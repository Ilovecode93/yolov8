from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('runs/segment/train3/weights/best.pt')
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO('/home/deepl/ultralytics_926revise/checkpoint/82/weights/best.pt')
# model = YOLO('checkpoint/520/best.pt')
# Train the model
model.train(data='smallpig.yaml', epochs=1000, imgsz=1024, patience= 500, batch=4,)
