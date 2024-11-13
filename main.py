import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class_names = {0: "", 1: "Anorganik", 2: "B3", 3: "Kertas", 4: "Organik"}

num_classes = 6
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("faster_rcnn_coco.pth", map_location=torch.device('cpu')))

model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

cap = cv2.VideoCapture(0)

threshold = 0.5

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)

        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > threshold:
                x1, y1, x2, y2 = box
                class_name = class_names.get(label, "Unknown")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({score:.2f})", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Real-Time Object Detection", frame)

        #  Press q to stop detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

cap.release()
cv2.destroyAllWindows()
