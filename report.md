# Deepfake Voice

王博文 522031910633

## 1. 任务目标

本次实验的目标是利用深度学习技术对音频数据进行分类，区分真实和伪造的声音。我们将使用传统机器学习方法（如支持向量机`SVM`和逻辑回归`Logistic Regression`）以及深度学习方法（如卷积神经网络`CNN`）来实现这一目标。实验将包括数据预处理、模型训练和评估等步骤。

## 2. 传统机器学习（SVM，Logistic Regression）
### 2.1 `train1.py`的实现




### 2.2 `test1.py`的实现




### 2.3 结果分析

| 模型 | Accuracy | F1 值 | AUC 值 |
|------|----------|-------|-------|
| svm_linear_sf_mfcc.joblib | 85.12% | 79.59% | 0.8886 |
| svm_linear_scipy_mfcc.joblib | 85.25% | 79.73% | 0.8902 |
| svm_rbf_mfcc.joblib | 85.12% | 78.48% | 0.8728 |
| logreg_mfcc.joblib | 82.00% | 75.34% | 0.8947 |

标准化之后，

| 模型 | Accuracy | F1 值 | AUC 值 |
|------|----------|-------|-------|
| svm_linear_mel.joblib | 99.88% | 99.83% | 1.0000 |
| svm_rbf_mel.joblib | 99.88% | 99.83% | 1.0000 |
| logreg_mel.joblib | 99.62% | 99.50% | 1.0000 |




## 3. 深度学习（CNN）
### 3.1 `train2.py`的实现




### 3.2 `test2.py`的实现




### 3.3 结果分析

| 模型 | Accuracy | F1 值 | AUC 值 |
|------|----------|-------|-------|
| dl_mels40_epochs10.pth | 96.12% | 94.55% | 0.9999 |
| dl_mels128_epochs10.pth | 99.88% | 99.83% | 1.0000 |


观察`dl_mels128_epochs10.pth`的训练过程：

```
Epoch 1/9 Train Loss: 0.1702 Acc: 93.53% | Val Loss: 0.0819 Acc: 97.00%
Epoch 2/9 Train Loss: 0.0722 Acc: 97.39% | Val Loss: 0.3704 Acc: 96.94%
Epoch 3/9 Train Loss: 0.1038 Acc: 95.67% | Val Loss: 0.0404 Acc: 98.67%
Epoch 4/9 Train Loss: 0.0275 Acc: 98.86% | Val Loss: 0.0118 Acc: 99.83%
Epoch 5/9 Train Loss: 0.0016 Acc: 99.97% | Val Loss: 0.0130 Acc: 99.83%
Epoch 6/10 Train Loss: 0.0012 Acc: 99.99% | Val Loss: 0.0110 Acc: 99.78%
Epoch 7/10 Train Loss: 0.0003 Acc: 99.99% | Val Loss: 0.0111 Acc: 99.78%
Epoch 8/10 Train Loss: 0.0002 Acc: 99.99% | Val Loss: 0.0117 Acc: 99.67%
Epoch 9/10 Train Loss: 0.0001 Acc: 99.99% | Val Loss: 0.0128 Acc: 99.78%
Epoch 10/10 Train Loss: 0.0001 Acc: 99.99% | Val Loss: 0.0127 Acc: 99.78%
```

发现可能存在过拟合现象，验证集的损失和准确率在第6个epoch以后没有明显下降。

改变`num_epochs`，尝试减少训练轮数。

| 模型 | Accuracy | F1 值 | AUC 值 |
|------|----------|-------|-------|
| dl_mels128_epochs10.pth | 99.88% | 99.83% | 1.0000 |
| dl_mels128_epochs9.pth  | 99.12% | 98.84% | 0.9999 |
| dl_mels128_epochs6.pth | 99.25% | 98.99% | 1.0000 |
| dl_mels128_epochs5.pth | 98.12% | 97.56% | 0.9999 |


## 4. 总结
### 4.1 收获&心得


### 4.2 遇到的问题 & 解决方法


## 5. 课程建议

