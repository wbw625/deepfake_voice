# 实现深度学习算法训练模型，打印训练过程。train2.py与train1.py在同一个目录下。
# 进入your_code_dir目录，运行python train2.py 能够启动训练过程。

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
import scipy.io.wavfile

data_dir = './deep_voice/train'

def extract_feature(file_path, feature_type='mel', n_mfcc=20, n_mels=128):
    sr, y = scipy.io.wavfile.read(file_path)
    y = y.astype(np.float32)
    # 归一化到[-1, 1]，防止溢出
    if y.dtype == np.int16:
        y = y / 32768.0
    elif y.dtype == np.int32:
        y = y / 2147483648.0
    if y.ndim > 1:
        y = y.mean(axis=1)
    if feature_type == 'mel':
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        feat = np.log1p(feat)
    else:  # mfcc
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = feat.astype(np.float32)
    return feat


class AudioDataset(Dataset):
    def __init__(self, data_dir, feature_type='mel', n_mels=128):
        self.X = []
        self.y = []
        for label_name, label in [('real', 1), ('fake', 0)]:
            folder = os.path.join(data_dir, label_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith('.wav'):
                    fpath = os.path.join(folder, fname)
                    try:
                        feat = extract_feature(fpath, feature_type=feature_type, n_mels=n_mels)
                        self.X.append(feat)
                        self.y.append(label)
                    except Exception as e:
                        print(f"Error processing {fpath}: {e}")
        # Pad/truncate to fixed length for batching
        max_len = max([x.shape[1] for x in self.X])
        self.X = [np.pad(x, ((0,0),(0,max_len-x.shape[1])), mode='constant') if x.shape[1]<max_len else x[:,:max_len] for x in self.X]
        self.X = np.stack(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self, flatten_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    model_path = './models/dl_mels128_epochs10.pth'

    n_mels = 128
    batch_size = 16
    epochs = 10
    lr = 1e-3

    print("加载数据中...")
    full_dataset = AudioDataset(data_dir, feature_type='mel', n_mels=n_mels)
    # 划分训练集和验证集
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=full_dataset.y)
    train_X = full_dataset.X[train_idx]
    train_y = full_dataset.y[train_idx]
    val_X = full_dataset.X[val_idx]
    val_y = full_dataset.y[val_idx]

    class SubsetDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

    train_dataset = SubsetDataset(train_X, train_y)
    val_dataset = SubsetDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')

    X_sample = torch.tensor(full_dataset.X[0]).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        x = X_sample
        x = torch.relu(nn.Conv2d(1, 16, kernel_size=3, padding=1).to(device)(x))
        x = nn.MaxPool2d(2).to(device)(x)
        x = torch.relu(nn.Conv2d(16, 32, kernel_size=3, padding=1).to(device)(x))
        x = nn.MaxPool2d(2).to(device)(x)
        flatten_dim = x.view(x.size(0), -1).size(1)

    model = SimpleCNN(flatten_dim=flatten_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("训练中...")
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y_batch).sum().item()
                val_total += y_batch.size(0)
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    print(f"保存模型到 {model_path}")
    torch.save(model.state_dict(), model_path)
    print("训练完成，模型已保存。")


if __name__ == '__main__':
    train()