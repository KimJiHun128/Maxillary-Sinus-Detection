from ultralytics import YOLO
import numpy as np
# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model #batch32
model.train(data="sinus_data.yaml", epochs=500, patience=50, batch=32, imgsz=512, device=0)  # train the model 작성한 yaml파일을 넣기

metrics = model.val()  # evaluate model performance on the validation set

#test

# model = YOLO("runs/aug/nano_aug/detect/train/weights/best.pt")  # load a pretrained model
model = YOLO("runs/detect/train/weights/best.pt")  # load a pretrainepip install -r requirements.txtd model (recommended for training)

metrics = model.val(data='sinus_data.yaml', split = 'test', max_det = 2)
# metrics = model.val(data='challenge.yaml', max_det = 2)




# 클래스별 정밀도, 재현율, F1 스코어 등 계산
precision = metrics.box.p
recall = metrics.box.r
f1_score = metrics.box.f1

print("AP scores:",metrics.box.all_ap)
print("ap_class_index:",metrics.box.ap_class_index)
print("AP at IoU thresholds from 0.5 to 0.95 for all classes:",metrics.box.ap)

print("precision:",precision)
print("recall:",recall)
print("f1_score:",f1_score)
print("Mean precision:",metrics.box.mp)
print("Mean recall:",metrics.box.mr)
print("평균 F1 스코어:", np.mean(f1_score))

model.predict("dataset/1080_bbox_final/test/images", save=True, imgsz=512)

