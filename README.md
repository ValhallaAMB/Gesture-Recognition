# Gesture Recognition - Real-Time ASL Alphabet Translator

## 🧠 Intro & Goals

This project presents a real-time sign language recognition system designed to bridge communication gaps for the deaf and hard of hearing communities. Built as a standalone Python desktop application, the system transcribes American Sign Language (ASL) alphabet gestures into on-screen text, with optional text-to-speech capabilities. The main goals include:

* Achieving >99% accuracy in ASL alphabet recognition.
* Providing <20ms latency per frame using a TensorFlow Lite model.
* Offering a highly usable interface with adjustable parameters.

The system is aimed at education, accessibility, and enhancing human-computer interaction.

---

## 📋 System Requirements

### Functional Requirements

* Start/stop camera capture via GUI.
* Real-time ASL gesture recognition and transcription.
* Spell-checking and basic NLP processing.
* Optional audio output of detected words.
* Adjustable parameters (e.g., confidence threshold, detection speed).

### Non-Functional Requirements

* 30 FPS performance with ≤20ms detection latency.
* Desktop-only deployment (Windows/Linux/macOS).
* Low memory usage; no frame or log persistence.
* Graceful failure and recovery from camera disconnection.

---

## 🛠️ Technology Stack

### Languages & Frameworks

* Python 3.10/3.11
* PyQt5 (GUI Framework)

### Libraries & Tools

* **Computer Vision & Data Collection**: OpenCV, MediaPipe Hands, CVZone
* **Machine Learning**: TensorFlow Lite, Numpy, Pandas
* **Post-processing**: pyspellchecker, TextBlob
* **Text-to-Speech**: pyttsx3 (offline)
* **Environment Management**: venv/virtualenv, requirements.txt
* **Version Control**: Git, GitHub (main + feature branches)

---

## 📷 Data Collection & Preparation

* **Total images**: \~25,164 (27 classes: A-Z + "None")
* **Captured via**: Desktop webcam with CVZone & MediaPipe Hands
* **Image size**: 300x300 pixels
* **Augmentation**: Horizontal flips, rotations, contrast adjustment
* **Ethics**: Data only from team members; no identifiable data collected

---

## 🧩 Model Design (Training, Evaluation and Results)

### Training & Architecture

* **Architecture**: MLP with 3 Dense layers + Softmax classifier
* **Training Platform**: Google Colab (TPU v2-8)
* **Feature Extractor**: MediaPipe hand landmark model

### Hyperparameters

* Optimizer: Adam | LR: 0.001 | Batch Size: 32 | Epochs: 10
* Activation: ReLU | Dropout: 0.1 | BatchNorm enabled

### Evaluation Metrics

* Training Accuracy: 98.76%
* Validation Accuracy: 99.34%
* Test Accuracy: 99.58%
* F1 Score: 0.9945 | Precision: 0.9967 | Recall: 0.9953

---

## 🚀 Deployment

### Steps to Run Locally

1. Install Python 3.10.x or 3.11.x from [python.org](https://www.python.org/)
2. (Optional) Install Git
3. Clone repository:

   ```bash
   git clone https://github.com/ValhallaAMB/Gesture-Recognition.git
   ```
4. Navigate to project folder:

   ```bash
   cd Gesture-Recognition
   ```
5. Create virtual environment:

   ```bash
   py -m venv venv
   venv\Scripts\activate (Windows) or source venv/bin/activate (Unix)
   ```
6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
7. Run the application:

   ```bash
   python main.py
   ```

---

## 📚 References

- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
- TensorFlow Lite: https://www.tensorflow.org/lite
- OpenCV Library: https://opencv.org/
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/
- TextBlob: https://textblob.readthedocs.io/en/dev/
- Pyttsx3: https://pyttsx3.readthedocs.io/en/latest/
- CVZone Library: https://github.com/cvzone/cvzone

---

> This project demonstrates that real-time gesture recognition can be fast, accessible, and highly accurate using lightweight, on-device ML techniques. Future plans include dynamic gesture detection, mobile/web support, and multilingual capabilities.

---

**Authors:** [Abdulmohaimin Bashir](https://github.com/ValhallaAMB), [Muhammad Bilal](https://github.com/Kou-hako), [Nusaibah Mekkaoui](https://github.com/NMekks)

**Advisor:** Prof. Dr. Alparslan Horasan

**Institution:** Istanbul Aydin University - Computer Engineering Department

**Date:** July 2024
