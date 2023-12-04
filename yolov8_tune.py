from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('/home/deepl/usedlocated_ultralytics/smallpig_weights/1110/weights/best.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='smallpig.yaml', use_ray=False, epochs=100, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)