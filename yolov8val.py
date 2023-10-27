from ultralytics import YOLO
#model=YOLO("checkpoint/625/weights/best.pt")
#model625 = model.val(data='coco128-seg.yaml')
#print("65 model map: ", model625.box.map)
#del model
model=YOLO("smallpig_weights/1016/weights/best.pt")
model524 = model.val(data='smallpig.yaml')
print("1016 model map: ",model524.box.map)
del model
#model=YOLO("checkpoint/82/weights/best.pt")
#model82 = model.val(data='coco128-seg.yaml')
#print("model82 model map: ",model82.box.map)
#del model
model=YOLO("runs/segment/train2/weights/best.pt")
model720 = model.val(data='smallpig.yaml')
print("1018 model map: ",model720.box.map)
