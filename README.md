__AI Gym Tracker__

A Flask web application that leverages deep learning to track and analyze gym exercises in real time.

__📋 Description__

AI Gym Tracker uses a pre-trained neural network to recognize exercise movements and provide instantaneous feedback through audio cues and visual indicators. It’s designed to help users refine their workout technique and monitor progress.

__✨ Key Features__

Real-Time Exercise Recognition: Detects and classifies exercises as you perform them.

Multimodal Feedback: Offers both audio prompts and on-screen visuals to guide users.

Lightweight Web Interface: Simple, responsive pages powered by Flask and Jinja2.

Customizable Model: Easily swap or retrain the underlying model stored in action.h5.

__🛠️ Tech Stack Overview__

Flask: Core web framework for routing and rendering templates.

TensorFlow & Keras: Deep learning libraries for model inference.

MediaPipe & OpenCV: Tools for capturing and processing video input.

NumPy & SciPy: Numerical computing support for data handling.

Matplotlib: Optional visualizations for development and debugging.

h5py: Efficient storage and retrieval of model weights in HDF5 format.
