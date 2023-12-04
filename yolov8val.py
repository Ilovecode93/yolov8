from ultralytics import YOLO
#model=YOLO("checkpoint/625/weights/best.pt")
#model625 = model.val(data='coco128-seg.yaml')
#print("65 model map: ", model625.box.map)
#del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1110/weights/best.pt")
model1110 = model.val(data='smallpig.yaml')
print("1110 model map: ",model1110.box.map)
del model
#model=YOLO("checkpoint/82/weights/best.pt")
#model82 = model.val(data='coco128-seg.yaml')
#print("model82 model map: ",model82.box.map)
#del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1029/weights/best.pt")
model1029 = model.val(data='smallpig.yaml')
print("1029 model map: ",model1029.box.map)
del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1115/weights/best.pt")
model1115 = model.val(data='smallpig.yaml')
print("1115 model map: ",model1115.box.map)
del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1117/weights/best.pt")
model1117 = model.val(data='smallpig.yaml')
print("1117 model map: ",model1117.box.map)
del model
model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1128/weights/best.pt")
model1128 = model.val(data='smallpig.yaml')
print("1128 model map: ",model1117.box.map)
del model
