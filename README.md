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

# 安装依赖
pip install -r requirement.txt
```

### 2. 下载预训练模型
从 [Google Drive](https://drive.google.com/drive/folders/1llKmPaOUhsAqxtopFdfnIBAlspvw_I4I?usp=drive_link) 下载 `UniLSeg20_no_finetune.pth` 并放置在项目根目录。

### 3. 基本使用
```bash
# 命令行使用
python demo.py --input_image_path real_sample_image.jpg --prompt 'white cloud' --output_image_path result.jpg
```
