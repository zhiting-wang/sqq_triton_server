from fastapi import FastAPI, HTTPException  
import numpy as np  
import torchvision.transforms as T
from io import BytesIO  
from PIL import Image  
import torch  
from tritonclient.http import InferenceServerClient, InferInput  
import httpx
import asyncio

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
        async with httpx.AsyncClient() as client:  
            response = await client.get(url)  
            response.raise_for_status()  # 若请求出错则抛出异常  
            img = Image.open(BytesIO(response.content)).convert("RGB")  # 转换为RGB格式  
            return img  
    except Exception as e:  
        print(f"下载失败: {e}")  
        # 处理下载失败的情况，返回默认图像  
        default_path = './16.jpg'  # 默认图像路径  
        return Image.open(default_path)  


@app.get("/")
async def root():
  return {"message": "Hello World"}


@app.post("/batch_count")  
async def batch_count(request: dict):  
    try:  
        # 解析请求体  
        images = request.get("images", [])  
        version = request.get("model", '')  
        max_batch_size = 32  # 与Triton配置保持一致  

        # # 批量下载和处理图像  
        # tensors = []  
        # for img_url in images[:max_batch_size]:  # 防止超过最大batch  
        #     img = await download_image(img_url)  # 下载图像  
        #     tensor = await process_image(img, version)  # 处理并转换为张量  
        #     tensors.append(tensor)  
        # 使用asyncio.gather并行下载和处理图像  
        download_tasks = [download_image(url) for url in images[:max_batch_size]]  
        downloaded_images = await asyncio.gather(*download_tasks)  

        # 过滤有效图像  
        valid_images = [img for img in downloaded_images if img is not None]  

        if not valid_images:  
            raise HTTPException(status_code=400, detail="No valid images provided")  

        # 并行处理图像  
        process_tasks = [process_image(img, version) for img in valid_images]  
        tensors = await asyncio.gather(*process_tasks)  
            
        # 创建批量张量  
        batch_tensor = torch.stack(tensors).to(device)  
        
        # Triton客户端请求  
        triton_client = InferenceServerClient(url=TRITON_URL)  
        inputs = [InferInput("input__0", batch_tensor.shape, "FP32")]  
        inputs[0].set_data_from_numpy(batch_tensor.cpu().numpy())  
        
        # 执行推理  
        result = triton_client.infer(model_name="fcn_model", inputs=inputs)  
        outputs = result.as_numpy("output__0")  
        
        # 后处理  
        return [{  
            "ball": max(0, min(int(round(o.item())), 16))  # 保持原有逻辑  
        } for o in outputs]  
        
    except ValueError as ve:  
        raise HTTPException(status_code=400, detail=str(ve))  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))  
