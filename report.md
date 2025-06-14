# Deepfake Voice

王博文 522031910633


## 1. 传统机器学习（SVM，Logistic Regression）
### 1.1 `train1.py`的实现




### 1.2 `test1.py`的实现




### 1.3 结果分析

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




## 2. 深度学习
### 2.1 `train2.py`的实现




### 2.2 `test2.py`的实现




### 2.3 结果分析


| 模型 | Accuracy | F1 值 | AUC 值 |
|------|----------|-------|-------|
| dl_model.pth | 96.12% | 94.55% | 0.9999 |



## 3.总结


