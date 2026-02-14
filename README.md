EMG-Based Muscle Fatigue Classification using 1D CNN
📌 Problem Statement

Muscle fatigue is a critical factor in:

Sports injury prevention

Rehabilitation monitoring

Workplace ergonomics

Human–robot interaction

Surface Electromyography (sEMG) provides a non-invasive way to monitor muscle activity, but detecting fatigue reliably is challenging because:

EMG signals are highly noisy and non-stationary

Fatigue progression varies between individuals

Traditional methods rely on handcrafted features and thresholds

Recent research shows that deep learning can automatically learn fatigue patterns, but:

Requires proper signal preprocessing

Needs robust labeling strategies

Suffers from dataset limitations

This project addresses these challenges by building a deep learning-based fatigue detection system using a real biomedical dataset.

The work is based on the publicly available dataset:

"A Comprehensive Dataset of Surface Electromyography and Self-Perceived Fatigue Levels for Muscle Fatigue Analysis" 

2b4e4a1e-1fe6-4f00-8639-a99e44c…

This dataset contains:

13 participants

13+ hours of sEMG recordings

Dynamic upper-limb movements

Self-perceived fatigue labels (3 levels)

🎯 Project Objective

The goal of this project is to:

Develop a robust deep learning system capable of automatically classifying muscle fatigue levels from raw multichannel EMG signals.

Specifically, the system predicts:

Class 0: Non-Fatigue

Class 1: Medium Fatigue

Class 2: High Fatigue

⚙️ Methodology
1️⃣ Signal Preprocessing

Steps performed:

Band-pass filtering (20–450 Hz)

Window segmentation (4-second windows)

50% overlap sliding windows

Transition region removal to avoid noisy labels

Z-score normalization per window

The 4-second window length aligns with typical contraction cycles and is widely used in fatigue analysis. 

2b4e4a1e-1fe6-4f00-8639-a99e44c…

2️⃣ Deep Learning Model

A 1D Convolutional Neural Network was designed to automatically learn temporal fatigue patterns.

Architecture:

Feature extractor:

Conv1D (4 → 32)

Conv1D (32 → 64)

Conv1D (64 → 128)

BatchNorm + ReLU + MaxPooling

Feature aggregation:

Adaptive Average Pooling

Flatten layer (128-dim feature vector)

Classifier:

Fully connected layer (128 → 64)

Dropout regularization

Output layer (3 classes)

Total trainable parameters:

≈ 45K parameters

This makes the model:

Lightweight

Fast to train

Suitable for real-time applications

3️⃣ Training Strategy

To ensure robustness:

Stratified train-validation-test split

Hyperparameter grid search

Learning rate scheduling

Early model checkpointing

Hyperparameters tuned:

Learning rate

Dropout rate

Batch size

📊 Results
Best Performance Achieved

Test Accuracy:

⭐ ~78%

Weighted F1-Score:

⭐ ~0.78

This is strong performance for:

Raw EMG classification

Subject-independent fatigue detection

Small biomedical dataset

🧪 Key Observations
What the model learned:

✔ Fatigue progression patterns
✔ Frequency and amplitude changes
✔ Temporal muscle activation trends

Major challenges:

Label subjectivity

Inter-subject variability

Limited dataset size

These are well-known challenges in EMG fatigue research. 

2b4e4a1e-1fe6-4f00-8639-a99e44c…

🚀 Contributions of This Project

This work provides:

🔬 Technical Contributions

Complete EMG preprocessing pipeline

Robust transition-aware window labeling

Efficient 1D CNN architecture for fatigue detection

Hyperparameter optimization framework

🧠 Research Contributions

Demonstrates feasibility of deep learning on perceived fatigue data

Shows reliable classification using minimal features

Provides reproducible experimental workflow

💻 Practical Contributions

Real-time capable fatigue detection system

Lightweight deployable model

Ready-to-use training and evaluation scripts
