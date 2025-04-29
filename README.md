# AI FITNESS COACH

**A Flask web application that leverages deep learning to track and analyze gym exercises in real time.**

---

## 📋 Description

**AI Fitness Coach** uses a pre-trained neural network to recognize exercise movements and provide instantaneous feedback through audio cues and visual indicators. It’s designed to help users refine their workout technique and monitor progress.

---

## ✨ Key Features

- **Real-Time Exercise Recognition**: Detects and classifies exercises as you perform them.
- **Multimodal Feedback**: Offers both audio prompts and on-screen visuals.
- **Lightweight Web Interface**: Simple, responsive pages powered by Flask and Jinja2.
- **Customizable Model**: Swap or retrain the neural network stored in `action.h5`.

---

## 🛠️ Tech Stack Overview

- **Flask** – Web framework for routing and rendering templates  
- **TensorFlow & Keras** – Deep learning libraries for model inference  
- **MediaPipe & OpenCV** – Video capture and pose estimation  
- **NumPy & SciPy** – Numerical computing support  
- **Matplotlib** – Development-time plotting and debugging  
- **h5py** – HDF5-based model weight storage 
