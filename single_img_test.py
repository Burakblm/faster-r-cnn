import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt


class_names = {0: "", 1: "Anorganik", 2: "B3", 3: "Kertas", 4: "Organik"}

num_classes = 6
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("faster_rcnn_coco.pth", map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# test image path
image_path = "/Users/burakbulama/Desktop/upwork projects/faster-rcnn/images/img1.jpg"


image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not loaded properly. Please check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(image_tensor)

# You can increase this value for more accurate detections.
threshold = 0.5

pred_boxes = predictions[0]['boxes'].cpu().numpy()
pred_scores = predictions[0]['scores'].cpu().numpy()
pred_labels = predictions[0]['labels'].cpu().numpy()

for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
    if score > threshold:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Class: {class_names[label]} Score: {score:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.figure(figsize=(12, 12)) 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
