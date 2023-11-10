from ultralytics import YOLO
#model=YOLO("checkpoint/625/weights/best.pt")
#model625 = model.val(data='coco128-seg.yaml')
#print("65 model map: ", model625.box.map)
#del model
model=YOLO("1029/weights/best.pt")
model524 = model.val(data='smallpig.yaml')
print("1029 model map: ",model524.box.map)
del model
#model=YOLO("checkpoint/82/weights/best.pt")
#model82 = model.val(data='coco128-seg.yaml')
#print("model82 model map: ",model82.box.map)
#del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1017/weights/best.pt")
model720 = model.val(data='smallpig.yaml')
print("1017 model map: ",model720.box.map)
