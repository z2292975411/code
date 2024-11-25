'''
@Project :yolov8
@File :run_DCNV3.py
@IDE :PyCharm
@Author :zsp
@Date :2024/4/15 20:22
'''
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
    model = YOLO('ultralytics/cfg/yolov8_SPPFCSPC.yaml').load('yolov8s.pt')
    # model = YOLO('yolov8s.pt')

    results = model.train(
        data='E:/datasets/Data/data.yaml',
        imgsz=640,
        epochs=100,
        # patience=50,
        batch=4,
        project='yolov8s',
        name='exp023_t',
        # cfg='./ultralytics/cfg/yolov8s_DCNV2.yaml'
    )