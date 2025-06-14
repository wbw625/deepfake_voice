# 加载训练好的深度学习模型，完成对测试集的测试，打印测试指标。test2.py与train1.py在同一个目录下。
# 进入your_code_dir目录，运行python test2.py 能够测试模型的指标。
# 为了便于测试，test2.py 打印结果到标准输出，最后一行是 ”Accuracy **%. F1 **%.”

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import scipy.io.wavfile
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

data_dir = './deep_voice/test'
model_path = 'dl_mel.pth'

def extract_feature(file_path, feature_type='mel', n_mfcc=20, n_mels=128):
    sr, y = scipy.io.wavfile.read(file_path)
    y = y.astype(np.float32)
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

def evaluate():
    n_mels = 128
    batch_size = 16

    print("加载测试集...")
    dataset = AudioDataset(data_dir, feature_type='mel', n_mels=n_mels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 用一条样本推理出flatten_dim
    X_sample = torch.tensor(dataset.X[0]).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        x = X_sample
        x = torch.relu(nn.Conv2d(1, 16, kernel_size=3, padding=1).to(device)(x))
        x = nn.MaxPool2d(2).to(device)(x)
        x = torch.relu(nn.Conv2d(16, 32, kernel_size=3, padding=1).to(device)(x))
        x = nn.MaxPool2d(2).to(device)(x)
        flatten_dim = x.view(x.size(0), -1).size(1)

    model = SimpleCNN(flatten_dim=flatten_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    print(f"Model: {model_path}")
    print(f"Accuracy {acc*100:.2f}%. F1 {f1*100:.2f}%. AUC {auc:.4f}")


if __name__ == '__main__':
    evaluate()