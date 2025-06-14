# 加载训练好的传统机器学习模型，完成对测试集的测试，打印测试指标。test1.py与train1.py在同一个目录下。
# 进入your_code_dir目录，运行python test1.py 能够测试模型的指标。
# 为了便于测试，test1.py 打印结果到标准输出，最后一行是 ”Accuracy **%. F1  **%.”

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.io.wavfile
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

data_dir = './deep_voice/test'


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


def test():
    # model_path = 'svm_linear_model.joblib'
    model_path = 'svm_rbf_model.joblib'
    # model_path = 'logreg_model.joblib'

    print("加载测试集...")
    X_test, y_test = load_data(data_dir, feature_type='mfcc')
    print(f"测试集样本数: {len(y_test)}")

    print("加载模型...")
    clf = joblib.load(model_path)

    print("预测中...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    print(f"Accuracy {acc*100:.2f}%. F1 {f1*100:.2f}%. AUC {auc:.4f}")


if __name__ == "__main__":
    test()
