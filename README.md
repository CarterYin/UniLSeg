# UniLSeg Demo 完整使用指南

## 📝 文件说明

- `demo.py` - 主要的演示程序
- `config/general_inference.yaml` - 配置文件（已更新路径）

#### 命令行接口
```bash
python demo.py --input_image_path /path/to/image --prompt "Object description" --output_image_path /path/to/output
```

## 使用方法

### 1. 环境准备
```bash
# 创建并激活conda环境
conda create -n UniLSeg python=3.7

conda activate UniLSeg

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y

# 安装依赖
pip install -r requirement.txt
```

### 2. 下载预训练模型
从 [Google Drive](https://drive.google.com/drive/folders/1llKmPaOUhsAqxtopFdfnIBAlspvw_I4I?usp=drive_link) 下载 `UniLSeg20_no_finetune.pth`文件和`pretrain`文件夹并放置在项目根目录（这里文件的具体目录是UniLSeg/UniLSeg/pretrain和UniLSeg/UniLSeg/UniLSeg20_no_finetune.pth）。

### 3.若无法直接下载到服务器，可以下载到本地，然后通过scp命令将文件传输到服务器。
- 对于文件夹：
```bash
scp -r /path/to/local/file user@server:/path/to/remote/directory
```
例如
```bash
scp -r ~/Desktop/my_folder/ yinchao@i****:/home/yinchao/
```

- 对于文件：
```bash
scp /path/to/local/file user@server:/path/to/remote/directory
```
例如
```bash
scp ~/Desktop/my_file.txt yinchao@i****:/home/yinchao/
```

### 4. 基本使用
```bash
# 命令行使用
python demo.py --input_image_path real_sample_image.jpg --prompt 'white cloud' --output_image_path result.jpg
```
