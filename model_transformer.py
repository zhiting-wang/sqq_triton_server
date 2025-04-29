import torch
import torchvision
import torch.nn as nn
from model import FCNModel

# 转换为torch_script模型

# 加载模型

# 导出为TorchScript  
device = 'cuda'
model = FCNModel().to(device)  
model.load_state_dict(torch.load('./weights/best_loss_model-1228.pth'))  
model.eval()  

# 使用样例输入跟踪模型  
example_input = torch.rand(1, 3, 960, 960).to(device)  
traced_script_module = torch.jit.trace(model, example_input)  
traced_script_module.save("fcn_model.pt") 

print("----------Finished Transformation------------")

# 验证输出维度  
# 创建符合预期的输入张量（batch_size=2）  
dummy_input = torch.rand(2, 3, 960, 960).to(device)  
output = model(dummy_input)  
assert output.shape == (2, 1), f"维度不匹配，实际输出维度：{output.shape}"  
print("✓ 模型验证通过")  


# 转换为onnx模型
'''
# 加载模型
model = FCNModel()
state_dict = torch.load('../weights/best_acc_model-994.pth')
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 960, 960)
input_names = [ "actual_input" ]
output_names = [ "output" ]

torch.onnx.export(model,
                 dummy_input,
                 "fcnmodel.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )


import onnxruntime as onnxrt

onnx_session= onnxrt.InferenceSession("resnet50.onnx")
onnx_inputs= {onnx_session.get_inputs()[0].name: torch.randn(1, 3, 224, 224).numpy()}
onnx_output = onnx_session.run(None, onnx_inputs)
img_label = onnx_output[0]

'''
