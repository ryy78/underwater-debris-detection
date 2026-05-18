from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = YOLO(r'D:/experiments/yolo11/ultralytics-main/model structure/yolo11-WMFE_ESSamp.yaml', task='detect')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(data=r'D:/experiments/yolo11/ultralytics-main/datasets/data.yaml',
                imgsz=640,
                epochs=200,
                single_cls=False,
                batch=16,
                name='datasets/yolo11-WMFE-Shapeloss',
                workers=0,
                device='0'
                )

# from ultralytics import YOLOWorld
#
# # Load a pretrained YOLOv8s-worldv2 model
# model = YOLOWorld("yolov8s-worldv2.pt")
#
# # Train the model on the COCO8 dataset for 100 epochs
# results = model.train(data="dataset/data.yaml", epochs=200, imgsz=640)
