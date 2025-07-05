# Hand Gesture Recognition for ASL Alphabet

## Overview
This project implements a robust hand gesture recognition system for American Sign Language (ASL) alphabet using deep learning. The system extracts hand landmarks using MediaPipe, processes them through a 1D convolutional neural network, and classifies 26 ASL letters (A-Z) with high accuracy.

## Key Features
- MediaPipe Hand Tracking: Real-time hand landmark extraction
- 1D Convolutional Neural Network: Specialized architecture for landmark data
- HDF5 Dataset Format: Efficient storage of preprocessed landmarks
- Batch Normalization: Improves training stability and speed
- Dropout Regularization: Prevents overfitting
- Relative Coordinate System: Wrist-centered coordinates for translation invariance

## Key Components
- 1. **Dataset Preprocessing (convert_dataset.py)**
  - Extracts 21 hand landmarks per image using MediaPipe
  - Converts absolute coordinates to wrist-relative coordinates
  - Stores data in efficient HDF5 format
  - Automatically splits data into train/test sets (80/20)
  - Handles images with no detected hands

- 2. **CNN Model (model.py)**
  - Specialized 1D convolutional architecture for landmark data
  - Four convolutional blocks with increasing filters
  - Batch normalization after each convolutional layer
  - Global pooling instead of flattening for better feature extraction
  - Dropout regularization to prevent overfitting
  - Softmax output layer for 26-class classification

- 3. **Utilities (cnn_utils.py and my_utils.py)**
  - Mini-batch creation for efficient training
  - One-hot encoding for labels
  - Sigmoid and ReLU activation functions
  - Forward propagation implementation
  - Prediction utilities

## Applications
- Real-time ASL-to-text translation
- Educational tools for learning sign language
- Accessibility features in communication apps
- Gesture-based control systems
- Security authentication via hand gestures
