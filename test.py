from ultralytics import YOLO

model = YOLO("runs/n/train/weights/best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)

metrics = model.val(data='sinus_data.yaml', split = 'test', max_det = 2)
# metrics = model.val(data='challenge.yaml', max_det = 2)




# 클래스별 정밀도, 재현율, F1 스코어 등 계산
precision = metrics.box.p
recall = metrics.box.r
f1_score = metrics.box.f1
print("precision:",precision)
print("recall:",recall)
print("f1_score:",f1_score)
print("AP scores:",metrics.box.all_ap)
print("ap_class_index:",metrics.box.ap_class_index)
print("AP at IoU thresholds from 0.5 to 0.95 for all classes:",metrics.box.ap)
print("Mean precision:",metrics.box.mp)
print("Mean recall:",metrics.box.mr)

'''
# results = model("dataset/1080_bbox_final.yolov8/test/images", save = True)
model.predict("dataset/1080_bbox_final/test/images", save=True, imgsz=512)
'''

