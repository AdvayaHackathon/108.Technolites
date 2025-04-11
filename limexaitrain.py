import sys
import os
import numpy as np
import cv2
import torch
import tensorflow as tf
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from captum.attr import GuidedBackprop
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Constants
SEVERITY_LABELS = ["Severe", "Mild", "Moderate", "Proliferate_DR", "No_DR"]
REMEDIES = {
    "Severe": "Immediate medical intervention required. Consult an ophthalmologist.",
    "Mild": "Regular eye checkups and controlled diet advised.",
    "Moderate": "Monitor blood sugar levels and follow prescribed medications.",
    "Proliferate_DR": "Laser treatment or anti-VEGF injections might be necessary.",
    "No_DR": "No immediate concern. Maintain a healthy lifestyle."
}

MODEL_PATH = "models/retinal_mamba_cnn.h5"
XAI_MODEL_PATH = "models/retinal_mamba_torch.pth"

# Image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Dummy Torch Model (Replace with actual architecture)
class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Matplotlib Bar Chart Canvas
class SeverityGraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_confidences(self, confidences, labels):
        self.ax.clear()
        self.ax.bar(labels, confidences, color='darkblue')
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Confidence')
        self.ax.set_title('Prediction Confidence by Severity')
        self.draw()

# Main App Class
class DiabetesApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.xai_model = None
        self.load_models()
        self.initUI()

    def load_models(self):
        print("[INFO] Loading TensorFlow model...")
        try:
            self.model = load_model(MODEL_PATH)
            print("[SUCCESS] TensorFlow model loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load TensorFlow model: {e}")

        print("[INFO] Loading PyTorch model for XAI...")
        try:
            self.xai_model = SimpleTorchModel()
            self.xai_model.load_state_dict(torch.load(XAI_MODEL_PATH, map_location=torch.device('cpu')))
            self.xai_model.eval()
            print("[SUCCESS] PyTorch model loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch model: {e}")
            self.xai_model = None

    def initUI(self):
        self.setWindowTitle("Diabetes Detection with XAI")
        self.setGeometry(100, 100, 700, 700)
        self.setStyleSheet("background-color: #e6f2ff;")

        self.label = QLabel("Upload Retinal Image for Analysis", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #003366; font-size: 18px; font-weight: bold;")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(224, 224)
        self.image_label.setStyleSheet("border: 2px solid black;")

        self.button = QPushButton("Upload Image", self)
        self.button.setStyleSheet(
            "background-color: white; color: black; font-size: 14px; font-weight: bold; padding: 5px; border-radius: 5px;"
        )
        self.button.clicked.connect(self.upload_image)

        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: darkred; font-size: 14px; font-weight: bold;")

        self.remedy_label = QLabel("", self)
        self.remedy_label.setAlignment(Qt.AlignCenter)
        self.remedy_label.setStyleSheet("color: darkgreen; font-size: 14px; font-weight: bold;")

        self.graph_canvas = SeverityGraphCanvas(self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.button, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        layout.addWidget(self.remedy_label)
        layout.addWidget(self.graph_canvas)

        self.setLayout(layout)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")

        if file_name:
            self.image_label.setPixmap(QPixmap(file_name).scaled(224, 224))

            predicted_label, confidences = self.predict_with_model(file_name)
            self.result_label.setText(f"Diagnosis: {predicted_label}")
            self.remedy_label.setText(f"Remedy: {REMEDIES.get(predicted_label, 'Consult a doctor for more details.')}")

            if confidences is not None:
                self.graph_canvas.plot_confidences(confidences, SEVERITY_LABELS)

            if self.xai_model:
                self.explain_with_xai(file_name)

    def predict_with_model(self, image_path):
        if self.model is None:
            return "Model not loaded!", None

        img = preprocess_image(image_path)
        prediction = self.model.predict(img)[0]
        predicted_label = SEVERITY_LABELS[np.argmax(prediction)]
        return predicted_label, prediction

    def explain_with_xai(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(img).unsqueeze(0)
        tensor_image.requires_grad = True

        guided_backprop = GuidedBackprop(self.xai_model)
        try:
            attr = guided_backprop.attribute(tensor_image, target=0)
            attr = attr.squeeze().detach().numpy()
            attr = np.transpose(attr, (1, 2, 0))

            plt.figure(figsize=(3, 3))
            plt.imshow(img, alpha=0.4)
            plt.imshow(attr, cmap='jet', alpha=0.6)
            plt.axis("off")
            plt.title("XAI: Guided Backprop")
            xai_output_path = "xai_output.png"
            plt.savefig(xai_output_path)
            plt.close()

            self.image_label.setPixmap(QPixmap(xai_output_path).scaled(224, 224))
        except Exception as e:
            print(f"[ERROR] XAI explanation failed: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiabetesApp()
    window.show()
    sys.exit(app.exec_())
