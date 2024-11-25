'''
@Project :distill.py
@File :get_channel.py
@IDE :PyCharm
@Author :zsp
@Date :2024/10/4 19:23
'''
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO(r'D:\lian\Documents\backup\thesis\yolov8n_sppfcspc_mpdiou\yolov8n_sppfcspc_MPDIoU\weights\best.pt')

# 查看模型结构
print(model.model)
