import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
# from matplotlib import pyplot as plt
import torch

import torchvision.models as models
from torch import nn
from collections import OrderedDict
from torchvision import transforms

data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


base_dir = os.getcwd()
# Load gender model
gender_model_dir = os.path.join(base_dir, 'resnet_18_spoof_model_40.pt')

# Load model
model = models.resnet18()
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100, 2)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
model.fc = fc

model.load_state_dict(torch.load(gender_model_dir))
model.eval()


def show_webcam(camera_number):
    mtcnn = MTCNN()
    classes = ('spoof', 'real')
    idx_to_class = {i: j for i, j in enumerate(classes)}
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv2)
        try:
            # detect face
            boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
            x, y, x2, y2 = [int(x) for x in boxes[0]]
            margin_x = int((x2-x)*0.1)
            margin_y = int((y2 -y) * 0.2)
            x -= margin_x;  x2 += margin_x; y -= margin_y;  y2 += margin_y;
            cropped_face = img_cv2[x:x2, y:y2]
            # liveness model
            tr_img = data_transform(cropped_face).float()
            tr_img = tr_img.unsqueeze(0)
            liveness = model(tr_img)
            liveness = liveness.argmax().item()
            liveness = idx_to_class[liveness]
            # draw results
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), thickness=2)
            img = cv2.putText(img, str(liveness), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)

        except Exception as e:
            print(e)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()
