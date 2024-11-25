'''
@Project :yolov8
@File :test_cuda.py
@IDE :PyCharm
@Author :zsp
@Date :2024/3/7 20:49
'''
import torch


print('pytorch_version:',torch.__version__)
print('is_cuda:',torch.cuda.is_available())
print('gpu_count:',torch.cuda.device_count())
print('cudnn_version:',torch.backends.cudnn.version())
print('cuda_version:',torch.version.cuda)
'''
pytorch_version: 2.2.1
is_cuda: True
gpu_count: 1
cudnn_version: 8700
cuda_version: 11.8
'''