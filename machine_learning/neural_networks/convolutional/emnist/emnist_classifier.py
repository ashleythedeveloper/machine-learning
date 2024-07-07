import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from os.path import join

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 47)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

models_path = 'models'
model = Net().to(device)
model.load_state_dict(torch.load(join(models_path, 'emnist_model_2024-07-02_19-44-52.pth'), map_location=device))
model.eval()

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    normalized = resized.astype(np.float32) / 255.0
        
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor, normalized

drawing = np.zeros((640, 640), dtype=np.uint8)
last_pos = None

# Modify the drawing function to use thicker lines
def draw(event, x, y, flags, param):
    global drawing, last_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        last_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        if last_pos:
            cv2.line(drawing, last_pos, (x, y), (255), 40)
            last_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        last_pos = None

cv2.namedWindow('EMNIST Classifier')
cv2.setMouseCallback('EMNIST Classifier', draw)

# Mapping of class indices
class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

pred = None
confidence = None
last_process_time = 0
process_interval = 0.4

while True:
    frame = drawing.copy()
    current_time = time.time()
    
    if current_time - last_process_time > process_interval:
        preprocessed, display_image = preprocess_image(frame)
        with torch.no_grad():
            output = model(preprocessed.to(device))
            pred = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()
        last_process_time = current_time

        cv2.imshow('Preprocessed', cv2.resize(display_image, (280, 280), interpolation=cv2.INTER_NEAREST))

    pred_text = f"Prediction: {class_mapping[pred]}" if pred is not None else "Prediction: N/A"
    cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    
    conf_text = f"Confidence: {confidence:.2f}" if confidence is not None else "Confidence: N/A"
    cv2.putText(frame, conf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    
    cv2.putText(frame, "Press 'c' to clear", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200), 2)

    cv2.imshow('EMNIST Classifier', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        drawing = np.zeros((480, 640), dtype=np.uint8)
        pred = None
        confidence = None

cv2.destroyAllWindows()