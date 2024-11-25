import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
from ultralytics.models.yolo.segment.distill import SegmentationDistiller
from ultralytics.models.yolo.pose.distill import PoseDistiller
from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        # 'model': r'C:\Users\lian\PycharmProjects\ultralytics\yolov8n\yolov8n_v2_DCNv2_cspc2\weights\best.pt',
        # 'model': r'C:\Users\lian\PycharmProjects\ultralytics\yolov8n\yolov8n_v2_DCNv22\weights\best.pt',
        # 'model': r'C:\Users\lian\PycharmProjects\runs\detect\train19\weights\best.pt',
        'model': r'C:\Users\lian\PycharmProjects\ultralytics-prune\ultralytics\runs\prune\yolov8n-sppfcspc_MPDIoU_2.5-lamp-exp3-finetune\weights\best.pt',
        # 'model': r'C:\Users\lian\PycharmProjects\ultralytics-prune\ultralytics\runs\prune\yolov8n-sppfcspc_MPDIoU-lamp-exp3-finetune2\weights\best.pt',
        'data': r'E:\datasets\my_data\Data_v3\data.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 8,
        'workers': 8,
        'cache': True,
        # 'optimizer': 'SGD',
        'optimizer': 'auto',
        'device': '0',
        'close_mosaic': 10,
        'project':'runs/distill',
        'name':'yolov8n_v3_SPPFCSPC_MPDIoU_2.5_s_mgd',
        
        # distill
        'prune_model': True,
        'teacher_weights': r'C:\Users\lian\PycharmProjects\runs\detect\train24\weights\best.pt',
        'teacher_cfg': 'yolov8s.pt',
        # 'teacher_weights': r'C:\Users\lian\PycharmProjects\runs\runs\detect\train2\weights\best.pt',
        # 'teacher_cfg': 'yolov8m.pt',
        # 'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8.yaml',
        'kd_loss_type': 'feature',
        # 'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        # 'logical_loss_type': 'l1',
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '12,15,18,21',
        'student_kd_layers': '12,15,18,21',
        # 'feature_loss_type': 'mimic',
        # 'feature_loss_type': 'cwd',
        'feature_loss_type': 'mgd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()