# AI Gym Tracker 🏋️‍♂️

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/Mediapipe-Enabled-green)


**AI Gym Tracker** is a multimodal fitness web application that uses real-time pose estimation and deep learning to track exercises, count repetitions, and evaluate workout performance using a webcam interface. It combines gesture recognition, audio-visual feedback, and intelligent classification for an interactive workout experience.

---

## ✨ Key Features

- 🎯 **Pose Detection** — Real-time tracking with MediaPipe.
- 🔁 **Rep Counting** — Angle-based rep detection using joint landmarks.
- ✅ **Performance Evaluation** — Feedback like *Perfect*, *Good*, or *Too Quick*.
- 🔥 **Calorie Estimation** — Estimates calories burned.
- 🔊 **Audio-Visual Feedback** — Multimodal guidance with sound and visuals.
- 🌐 **Web Interface** — Accessible via browser, no install needed.

---

## 🧠 Tech Stack

| Layer         | Technology                            |
|---------------|----------------------------------------|
| Frontend      | HTML5, Bootstrap                      |
| Backend       | Flask (Python)                        |
| Machine Learning | LSTM model (Keras)                  |
| Computer Vision | OpenCV, MediaPipe                   |

---

## 📂 Project Structure

├── static/ # Assets (images, audio)
├── templates/ # HTML templates
├── app.py # Main Flask app
├── action.h5 # Trained LSTM model
├── requirements.txt # Dependencies
└── README.md # Documentation

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone <repo_url>
cd <repo_directory>

2. Set Up Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the App
python app.py

✅ Requirements

**Python 3.7+**

**Webcam-enabled device**

**Chrome or any modern browser**

**Internet connection for model inference**

