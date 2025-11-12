 Wild Animal Detection using YOLOv8

## 📘 Overview
This project focuses on **wild animal detection using YOLOv8**, a state-of-the-art deep learning model for object detection.  
The system accurately identifies and classifies wild animals such as **elephants, lions, bears, and zebras** from images or live video streams.  
It enables **real-time wildlife monitoring**, safety alerts, and conservation-related applications.

---

## 🚀 Features
- Real-time wild animal detection using YOLOv8
- High accuracy and fast inference
- Easy deployment in Flask or OpenCV
- Customizable model (`best.pt`) for different animal classes

---

## 🧠 Model Training
The model is trained in **Google Colab** using the **Ultralytics YOLOv8 framework**.  
It uses a curated dataset of wild animals to ensure accurate classification.

### Steps:
1. Install dependencies:
   ```bash
   !pip install ultralytics


Train the model:

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=50, imgsz=640)


The trained model best.pt will be saved under:

runs/detect/animal_detector/weights/best.pt

🖥️ Real-Time Detection (VS Code)

Once training is complete, move best.pt to your Visual Studio Code project and run:

from ultralytics import YOLO
model = YOLO('best.pt')
model.predict(source=0, show=True)  # 0 for webcam

📁 Project Structure
├── dataset/
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── runs/
│   └── detect/animal_detector/
│       └── weights/best.pt
├── app.py  # Flask or OpenCV real-time detection
└── README.md

📊 Results

The trained YOLOv8 model achieves high detection accuracy for multiple wild animal species in diverse environments.

📜 License

This project is open-source and available under the MIT License.

👩‍💻 Author

Geethu B Raj
MTech AI & ML | Wildlife Monitoring Research
