# 实现传统机器学习算法训练模型，打印训练过程。
# 数据集目录默认在train1.py的上一级目录中，目录关系如下所示。
# 进入your_code_dir目录，运行python train1.py 能够启动训练过程。

import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import soundfile as sf
import scipy.io.wavfile

data_dir = './deep_voice/train'


# def extract_feature(file_path, feature_type='mfcc', n_mfcc=20):
#     y, sr = sf.read(file_path)
#     if y is None or len(y) == 0:
#         raise ValueError("Empty audio file")
#     if y.ndim > 1:
#         y = y.mean(axis=1)
#     y = y.astype(np.float32)

#     if feature_type == 'cqt':
#         feat = librosa.cqt(y, sr=sr)
#         feat = np.abs(feat)
#         return np.mean(feat, axis=1)
#     elif feature_type == 'mel':
#         feat = librosa.feature.melspectrogram(y, sr=sr)
#         feat = np.log1p(feat)
#         return np.mean(feat, axis=1)
#     else:  # mfcc
#         feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#         return np.mean(feat, axis=1)


def extract_feature(file_path, feature_type='mfcc', n_mfcc=20):
    sr, y = scipy.io.wavfile.read(file_path)
    y = y.astype(np.float32)
    # 归一化到[-1, 1]，防止溢出
    if y.dtype == np.int16:
        y = y / 32768.0
    elif y.dtype == np.int32:
        y = y / 2147483648.0
    if y.ndim > 1:
        y = y.mean(axis=1)
    if feature_type == 'cqt':
        feat = librosa.cqt(y, sr=sr)
        feat = np.abs(feat)
        return np.mean(feat, axis=1)
    elif feature_type == 'mel':
        feat = librosa.feature.melspectrogram(y, sr=sr)
        feat = np.log1p(feat)
        return np.mean(feat, axis=1)
    else:  # mfcc
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(feat, axis=1)

def load_data(data_dir, feature_type='mfcc'):
    X, y = [], []
    for label_name, label in [('real', 1), ('fake', 0)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                fpath = os.path.join(folder, fname)
                try:
                    feat = extract_feature(fpath, feature_type=feature_type)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    return np.array(X), np.array(y)


def train_svm_linear():
    model_path = 'svm_linear_model.joblib'
    print("加载数据中...")
    X, y = load_data(data_dir, feature_type='mfcc')  # 可选 'cqt' 或 'mel'
    print(f"数据量: {len(y)}")

    print("训练SVM模型...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    print(f"保存模型到 {model_path}")
    joblib.dump(clf, model_path)
    print("训练完成，模型已保存。")


def train_svm_rbf():
    model_path = 'svm_rbf_model.joblib'
    print("加载数据中...")
    X, y = load_data(data_dir, feature_type='mfcc')  # 可选 'cqt' 或 'mel'
    print(f"数据量: {len(y)}")

    print("训练SVM模型...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X, y)

    print(f"保存模型到 {model_path}")
    joblib.dump(clf, model_path)
    print("训练完成，模型已保存。")


def train_logistic_regression():
    model_path = 'logreg_model.joblib'
    print("加载数据中...")
    X, y = load_data(data_dir, feature_type='mfcc')  # 可选 'cqt' 或 'mel'
    print(f"数据量: {len(y)}")

    print("训练逻辑回归模型...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    print(f"保存模型到 {model_path}")
    joblib.dump(clf, model_path)
    print("训练完成，模型已保存。")


if __name__ == '__main__':
    # train_svm_linear()
    train_svm_rbf()
    # train_logistic_regression()