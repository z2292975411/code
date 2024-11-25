'''
@Project :distill.py
@File :test.py
@IDE :PyCharm
@Author :zsp
@Date :2024/7/31 17:47
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook



model = YOLO('yolov8n.pt')  # 使用你需要的模型权重文件

# 查看模型结构，找到你感兴趣的层的名称
print(model)

# 假设我们选择模型中的某一层，比如 backbone 中的某一层
# layer_name = 'model.model.Conv'  # 根据实际模型的层名称
# 获取模型的Sequential模块
sequential_module = model.model.model
# 注册钩子函数到索引为2的C2f模块和索引为9的SPPF模块
sequential_module[21].register_forward_hook(get_activation('C2f_21'))
sequential_module[9].register_forward_hook(get_activation('SPPF_9'))

# 遍历并打印每一层的索引和名称
for idx, layer in enumerate(sequential_module):
    print(f"Layer {idx}: {layer}")

# layer = dict(model.named_modules())[layer_name]
# layer.register_forward_hook(get_activation('C2f'))

# 加载并预处理输入图像
input_image = Image.open(r"E:\datasets\my_data\Data_v3\test\images\a09790cfda8dcdc8.jpg")
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

# 将图像传入模型
model(input_tensor)

# 获取激活图
activation_c2f = activations['C2f_21']
activation_sppf = activations['SPPF_9']

# 将激活图转换为 NumPy 数组
activation_c2f = activation_c2f.squeeze().cpu().numpy()
activation_sppf = activation_sppf.squeeze().cpu().numpy()

# 可视化多个通道的激活图
def plot_activation_maps(activation, num_channels=4):
    # plt.plot(num_channels)
    for i in range(num_channels):
        # ax = axes[i]
        plt.imshow(activation[i])
        plt.axis('off')
        # ax.set_title(f'Channel {i}')
    plt.show()
# 可视化特定通道的激活图
def plot_specific_channel(activation, channel_idx):
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(activation[channel_idx], cmap='viridis')
    ax.axis('off')  # 关闭坐标轴
    # 调整图像显示，确保无多余画布
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

# 可视化索引为2的C2f模块的第0个通道的激活图
plot_specific_channel(activation_c2f, channel_idx=128)

# 可视化索引为9的SPPF模块的第0个通道的激活图
# plot_specific_channel(activation_sppf, channel_idx=0)

# 可视化索引为2的C2f模块的前4个通道的激活图
# plot_activation_maps(activation_c2f, num_channels=4)

# 可视化索引为9的SPPF模块的前4个通道的激活图
# plot_activation_maps(activation_sppf, num_channels=4)