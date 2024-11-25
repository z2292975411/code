'''
@Project :yolov8
@File :run.py
@IDE :PyCharm
@Author :zsp
@Date :2024/3/4 18:17
'''
import multiprocessing

from ultralytics import YOLO
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 在这里放置你的主程序代码

    # Create a new YOLO model from scratch
    # model = YOLO('C:/Users/lian/PycharmProjects/yolov8/datasets/yolov8.yaml')

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('yolov8n.pt')
    # model = YOLO('C:/Users/lian/PycharmProjects/runs/detect/train3/weights/last.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # results = model.train(data='C:/Users/lian/PycharmProjects/yolov8/datasets/data.yaml', epochs=500,device=0)

    # Evaluate the model's performance on the validation set
    # results = model.val()

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')
    # results = model.predict(source='C:/Users/lian/PycharmProjects/yolov8/ultralytics/assets/bus.jpg',save=True)
    # Export the model to ONNX format
    # success = model.export(format='onnx')
    # model = YOLO('yolov8s.pt').load('../ultralytics/yolov8s/train/200epoch/exp01_yolov8s/weights/last.pt')
    # model = YOLO('./ultralytics/cfg/yolov8s_L_FFCA.yaml').load(r'C:\Users\lian\PycharmProjects\yolov8_test\ultralytics\yolov8s\train\exp0120\weights\last.pt')
    # model = YOLO('./ultralytics/cfg/yolov8n.yaml').load('yolov8n.pt')
    # model = YOLO('./ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8s.pt')
    # model = YOLO('../ultralytics/yolov8s/train/200epoch/exp01_yolov8s/weights/last.pt')
    model = YOLO(r'C:\Users\lian\PycharmProjects\yolov8\ultralytics\cfg\yolov8_SimSPPF.yaml').load('yolov8n.pt')
    results = model.train(
        data='E:/datasets/my_data/Data_v3/data.yaml',
        resume=True,
        imgsz=640,
        epochs=250,
        # patience=50,
        batch=16,
        project='./yolov8n/train/',
        name='yolov8n_SimSPPF',
        conf=0.5
    )