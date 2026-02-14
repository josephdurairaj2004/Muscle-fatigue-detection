# ============================================================
# FINAL 1D-CNN Fatigue Classification (FULL TUNING VERSION)
# - 4s windows
# - Transition filtering
# - Hyperparameter tuning
# - 120 epochs per configuration
# - Best model saving (validation-based)
# - Final test evaluation
# - Confusion matrix
# - Save predictions + test set
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================= CONFIG =================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EMG_DIR = os.path.join(PROJECT_ROOT, "subject_emg")
LABEL_DIR = os.path.join(PROJECT_ROOT, "unzip_file")

FS = 1259
LABEL_FS = 50

WINDOW = 4 * FS
STRIDE = WINDOW // 2
TRANSITION_MARGIN = 4

EPOCHS = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameter grid
LR_LIST = [1e-3, 5e-4, 3e-4, 1e-4]
DROPOUT_LIST = [0.3, 0.4, 0.5]
BATCH_LIST = [8, 16, 32]

# ================= FILTER =================

def bandpass_filter(x):
    b, a = signal.butter(4, (20, 450), btype="bandpass", fs=FS)
    return signal.filtfilt(b, a, x, axis=1)

# ================= LOAD DATA =================

X_all, y_all, trial_ids = [], [], []

for subj in os.listdir(EMG_DIR):

    emg_path = os.path.join(EMG_DIR, subj)
    label_path = os.path.join(LABEL_DIR, subj)

    if not os.path.isdir(emg_path) or not os.path.isdir(label_path):
        continue

    for file in os.listdir(emg_path):

        if not file.endswith(".csv") or "mvc" in file.lower():
            continue

        emg_file = os.path.join(emg_path, file)
        label_file = os.path.join(label_path, file)

        if not os.path.exists(label_file):
            continue

        emg_df = pd.read_csv(emg_file)
        emg_cols = [c for c in emg_df.columns if "EMG" in c]
        if len(emg_cols) < 4:
            continue

        emg = emg_df[emg_cols[:4]].to_numpy().T
        emg = np.nan_to_num(emg)
        emg = bandpass_filter(emg)

        lab_df = pd.read_csv(label_file)
        labels = lab_df["label"].values

        change_indices = np.where(np.diff(labels) != 0)[0]
        change_times = change_indices / LABEL_FS

        total_samples = emg.shape[1]

        for start in range(0, total_samples - WINDOW, STRIDE):

            end = start + WINDOW
            win = emg[:, start:end]

            mid_sample = start + WINDOW // 2
            mid_time = mid_sample / FS

            if any(abs(mid_time - ct) < TRANSITION_MARGIN for ct in change_times):
                continue

            label_idx = int(mid_time * LABEL_FS)
            if label_idx >= len(labels):
                continue

            win = (win - win.mean(axis=1, keepdims=True)) / \
                  (win.std(axis=1, keepdims=True) + 1e-6)

            X_all.append(win.astype(np.float32))
            y_all.append(int(labels[label_idx]))
            trial_ids.append(f"{subj}_{file}")

X_all = np.array(X_all)
y_all = np.array(y_all)
trial_ids = np.array(trial_ids)

print("Total windows:", X_all.shape)

# ================= SPLIT =================

X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
    X_all, y_all, trial_ids,
    test_size=0.3,
    stratify=y_all,
    random_state=42
)

X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
    X_temp, y_temp, id_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

np.save("saved_test_X.npy", X_test)
np.save("saved_test_y.npy", y_test)

# ================= DATASET =================

class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ================= MODEL =================

class CNN1D(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= HYPERPARAMETER SEARCH =================

best_val_acc = 0
best_config = None

for lr in LR_LIST:
    for dropout in DROPOUT_LIST:
        for batch_size in BATCH_LIST:

            print(f"\nTraining: LR={lr}, Dropout={dropout}, Batch={batch_size}")

            train_loader = DataLoader(
                EMGDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = DataLoader(
                EMGDataset(X_val, y_val),
                batch_size=batch_size,
                shuffle=False
            )

            model = CNN1D(dropout).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )

            for epoch in range(EPOCHS):

                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_preds, val_trues = [], []

                with torch.no_grad():
                    for xb, yb in val_loader:
                        out = model(xb.to(DEVICE))
                        val_preds.extend(out.argmax(1).cpu().numpy())
                        val_trues.extend(yb.numpy())

                val_acc = accuracy_score(val_trues, val_preds)
                scheduler.step(val_acc)

            print("Validation Accuracy:", val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (lr, dropout, batch_size)
                torch.save(model.state_dict(), "best_model.pth")

print("\nBest Config:", best_config)
print("Best Validation Accuracy:", best_val_acc)

# ================= FINAL TEST =================

lr, dropout, batch_size = best_config

test_loader = DataLoader(
    EMGDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False
)

best_model = CNN1D(dropout).to(DEVICE)
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.eval()

test_preds, test_trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        out = best_model(xb.to(DEVICE))
        test_preds.extend(out.argmax(1).cpu().numpy())
        test_trues.extend(yb.numpy())

test_preds = np.array(test_preds)
test_trues = np.array(test_trues)

raw_acc = accuracy_score(test_trues, test_preds)

print("\nRaw Test Accuracy:", raw_acc)
print("\nConfusion Matrix:")
print(confusion_matrix(test_trues, test_preds))
print("\nClassification Report:")
print(classification_report(test_trues, test_preds, digits=4))

np.save("y_test.npy", y_test)
np.save("y_pred_raw.npy", test_preds)
