from fastapi import FastAPI, HTTPException  
import numpy as np  
import torchvision.transforms as T
from io import BytesIO  
from PIL import Image  
import torch  
from tritonclient.http import InferenceServerClient, InferInput  
import httpx
import asyncio
import time
from collections import deque

app = FastAPI()  
TRITON_URL = "localhost:8000"  
device = 'cuda'
IMG_W = 960  
target_width = 1280  # 目标高度
target_height = 720  # 目标宽度
padding_color = (0, 0, 0)  # 黑色填充  

async def process_image(img: Image.Image, version: str) -> torch.Tensor:  
    """对图像进行处理，包括添加黑边和调整尺寸等."""  
    if 'V30' in version:  # V30锁球器摄像头  
        img_array = np.array(img)  
        top_black_height = img_array.shape[0] // 4  
        img_array[:top_black_height, :] = [0, 0, 0]  # 添加黑色区域  
        img = Image.fromarray(img_array)  
    elif img.width == 1600:  # V20锁球器摄像头  
        new_height = target_height  
        new_width = int(img.width * (target_height / img.height))  
        img = img.resize((new_width, new_height))  
        new_img = Image.new("RGB", (target_width, target_height), padding_color)  
        new_img.paste(img, ((target_width - new_width) // 2, (target_height - new_height) // 2))  
        img = new_img  

    # 转换为张量  
    transform = T.Compose([  
        T.Resize([IMG_W, IMG_W]),  
        T.ToTensor(),  
    ])  
    img_tensor = transform(img)  
    return img_tensor    

async def download_image(url: str) -> Image.Image:  
    """异步下载图像并返回PIL格式的图像对象."""  
    try:  
        async with httpx.AsyncClient(timeout=8.0) as client:  
            try:
                response = await client.get(url)  
                response.raise_for_status()  # 若请求出错则抛出异常  
                img = Image.open(BytesIO(response.content)).convert("RGB")  # 转换为RGB格式  
                return img  
            except httpx.TimeoutException as timeout_error:
                print(f"下载超时: {url}, 错误: {timeout_error}")
                # 超时错误处理
                default_path = './16.jpg'  # 默认图像路径
                print(f"使用默认图像替代: {default_path}")
                return Image.open(default_path)
    except Exception as e:  
        print(f"下载失败: {e}, URL: {url}")  
        # 处理下载失败的情况，返回默认图像  
        default_path = './16.jpg'  # 默认图像路径  
        return Image.open(default_path)  


@app.get("/")
async def root():
  return {"message": "Hello World"}


@app.post("/batch_count")  
async def batch_count(request: dict):  
    # 记录开始时间  
    start_time = time.time()  
    try:  
        # 解析请求体  
        images = request.get("images", [])  
        version = request.get("model", '')  
        max_batch_size = 16  # 与Triton配置保持一致 

        print(f"收到请求，包含 {len(images)} 张图像，模型版本: {version}")  

        # 使用asyncio.gather并行下载和处理图像  
        download_tasks = [download_image(url) for url in images[:max_batch_size]]  
        downloaded_images = await asyncio.gather(*download_tasks)  

        print(f"已下载 {len(downloaded_images)} 张图像")

        # 过滤有效图像  
        valid_images = []
        valid_urls = []  
        for i, img in enumerate(downloaded_images):  
            if img is not None:  
                valid_images.append(img)  
                valid_urls.append(images[i])  

        print(f"有效图像数量: {len(valid_images)}")

        if not valid_images:  
            raise HTTPException(status_code=400, detail="No valid images provided")  

        # 并行处理图像  
        print("开始处理图像...")
        process_tasks = [process_image(img, version) for img in valid_images]  
        tensors = await asyncio.gather(*process_tasks)  
            
        # 创建批量张量  
        batch_tensor = torch.stack(tensors).to(device)  
        
        # Triton客户端请求  
        triton_client = InferenceServerClient(
            url=TRITON_URL,
            concurrency=4,  # 并发请求数
            connection_timeout=10.0,  # 连接超时
            network_timeout=60.0  # 网络超时
            )  
        inputs = [InferInput("input__0", batch_tensor.shape, "FP32")]  
        inputs[0].set_data_from_numpy(batch_tensor.cpu().numpy())  
        
        # 执行推理  
        print("开始执行推理...")
        result = triton_client.infer(model_name="fcn_model", inputs=inputs)  
        outputs = result.as_numpy("output__0")  
        print(f"推理完成，获得 {len(outputs)} 个结果")

        # 计算处理时间并记录
        end_time = time.time()
        processing_time = end_time - start_time
        # 打印单次接口处理耗时
        print(f"接口调用完成，耗时: {processing_time:.3f}秒, ")

        
        # 修改后处理，返回包含ball和image的字典  
        return [{  
            "ball": max(0, min(int(round(o.item())), 16)),  # 保持原有逻辑  
            "image": url  # 添加对应的图像URL  
        } for o, url in zip(outputs, valid_urls)]  
        
    except ValueError as ve:  
        raise HTTPException(status_code=400, detail=str(ve))  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))  
