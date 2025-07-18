# deepfake_voice
DeepFake Voice Recognition

(2024-2025-2)-NIS3363-01-机器学习


## 1. 环境配置

本项目使用Python 3.8.18版本，系统为Ubuntu 20.04.1 LTS，x86_64架构。

使用如下的安装脚本：

```bash
ENV_NAME=voice
# 创建虚拟环境并激活
conda create --name voice python=3.8 -y
conda activate voice

# 首先安装torch相关的库
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 torchtext==0.11.1 cudatoolkit=10.2 -c pytorch -y
# 安装常见深度学习和机器学习的工具库
# 其中librosa用于提供音频相关的处理接口
conda install transformers==4.35.2 librosa==0.10.2 scipy==1.9.3 -c conda-forge -y

# conda和pip交替混用可能会导致一些版本兼容性的问题
# 因此使用conda安装完相关的库之后再使用pypi安装其他的要用到的环境
pip install x-transformers==0.15.0
```

但此时可能会出现一些版本兼容性的问题，需要更新一些库的版本：

```bash
pip uninstall numpy
pip install --upgrade librosa soundfile numpy
```


## 2. 传统机器学习

### 2.1 训练

```bash
python train1.py
```

### 2.2 测试

```bash
python test1.py
```


## 3. 深度学习

### 3.1 训练

```bash
python train2.py
```

### 3.2 测试

```bash
python test2.py
```


## 4. 额外说明

在各python脚本中，可能均需要根据实际情况修改数据集、模型的路径：

```python
data_dir = './deep_voice/train'
```

```python
model_path = './models/dl_mels128_epochs10.pth'
```


## 5. 模型文件说明

模型文件存放在`./models/`目录下。其中，"dl_"开头的文件为深度学习模型，其余为传统机器学习模型。

具体的模型选择、参数设置等在模型文件名中可见。
