EMG-Based Muscle Fatigue Classification Using Deep Learning
Overview

Muscle fatigue detection is a critical component in modern biomedical engineering applications, including rehabilitation monitoring, injury prevention, ergonomic assessment, and human–machine interaction. Surface electromyography (sEMG) provides a non-invasive mechanism to measure muscle activity; however, accurate fatigue classification from raw EMG signals remains challenging due to signal variability, noise, inter-subject differences, and subjective labeling.

This project presents a complete deep learning pipeline for automatic classification of muscle fatigue levels from raw multichannel EMG signals using a one-dimensional convolutional neural network (1D-CNN). The system processes raw EMG data, performs signal preprocessing and transition filtering, and learns fatigue-related temporal patterns without reliance on handcrafted features.

Problem Statement

Traditional EMG fatigue detection methods rely heavily on frequency-domain features such as median frequency and mean power frequency. These approaches often require manual feature engineering and struggle to generalize across subjects and movement conditions.

The objective of this project is to develop an automated, data-driven fatigue classification system capable of:

Learning discriminative fatigue patterns directly from raw EMG signals

Handling noisy and non-stationary biomedical time-series data

Providing robust classification across multiple participants and movement trials

The system aims to classify muscle fatigue into three levels:

Non-Fatigue

Moderate Fatigue

High Fatigue

Dataset Description

The model is trained and evaluated on a publicly available biomedical dataset containing surface electromyography recordings collected from healthy adult participants performing dynamic upper-limb movements.

Key dataset characteristics:

13 participants

Multichannel EMG recordings (4 muscles per arm)

Sampling frequency: 1259 Hz

Self-reported fatigue labels recorded at 50 Hz

Multiple movement protocols and exercise sessions

This dataset provides realistic conditions for fatigue classification, including inter-subject variability and subjective labeling uncertainty.

Methodology
Signal Preprocessing

Raw EMG signals undergo the following preprocessing steps:

Band-pass filtering between 20 Hz and 450 Hz to remove motion artifacts and noise

Segmentation into fixed-length time windows of four seconds

Fifty percent overlap between consecutive windows

Removal of transition regions near fatigue label changes to reduce label noise

Per-window normalization using z-score standardization

These preprocessing steps ensure high-quality input data while preserving temporal characteristics relevant to fatigue progression.

Model Architecture

A lightweight one-dimensional convolutional neural network is designed to extract temporal features from multichannel EMG signals.

Feature Extraction Layers

Conv1D (4 input channels → 32 filters, kernel size 7)

Conv1D (32 → 64 filters, kernel size 5)

Conv1D (64 → 128 filters, kernel size 3)

Batch normalization and ReLU activation after each convolution

Max pooling to reduce temporal dimensionality

Feature Aggregation

Adaptive average pooling to produce a fixed-length representation

Flattening into a 128-dimensional feature vector

Classification Head

Fully connected layer (128 → 64)

ReLU activation and dropout regularization

Output layer (64 → 3 classes)

The architecture contains approximately forty-five thousand trainable parameters, making it computationally efficient and suitable for real-time deployment.

Training Strategy

The model training pipeline includes:

Stratified dataset splitting into training, validation, and testing sets

Hyperparameter optimization using grid search

Learning rate scheduling based on validation performance

Model checkpointing using best validation accuracy

Hyperparameters tuned during training include learning rate, dropout rate, and batch size.

Results

The trained model achieves strong performance on the fatigue classification task:

Test accuracy: approximately 78 percent

Weighted F1 score: approximately 0.78

Balanced classification performance across fatigue levels

Considering the subjective nature of fatigue labels and inter-participant variability, this level of performance represents a reliable and competitive outcome for raw EMG classification.

Key Contributions

This project provides:

A complete end-to-end pipeline for EMG fatigue analysis

Robust preprocessing and transition filtering methodology

A lightweight and effective deep learning architecture for time-series classification

Reproducible experimental setup and evaluation workflow

The work demonstrates the feasibility of deep learning-based fatigue detection using real biomedical datasets.
