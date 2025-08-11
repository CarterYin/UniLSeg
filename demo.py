import argparse
import os
import warnings
from typing import Dict, Any
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from loguru import logger

import utils.config as config
from model import build_segmenter
from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
_tokenizer = _Tokenizer()


def tokenize(texts, context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    对输入文本进行tokenize处理
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def convert(img):
    """
    图像预处理：归一化
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()
    img.div_(255.).sub_(mean).div_(std)
    return img


def getTransformMat(img_size, input_size, inverse=False):
    """
    计算图像变换矩阵
    """
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    
    # 检查输入参数的有效性
    if ori_h <= 0 or ori_w <= 0 or inp_h <= 0 or inp_w <= 0:
        raise ValueError(f"无效的尺寸参数: img_size={img_size}, input_size={input_size}")
    
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None


def forward(model: Any, image: Image.Image, description: str) -> Dict:
    """
    核心处理函数：对单张图片进行处理
    
    Args:
        model: 预训练的UniLSeg模型
        image: PIL.Image.Image格式的输入图像
        description: 目标对象的描述文本
    
    Returns:
        Dict: 包含处理结果的字典，可以直接用于可视化
    """
    model.eval()
    
    # 将PIL图像转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ori_img = img_cv.copy()
    
    # 图像预处理
    img_size = img_cv.shape[:2]
    input_size = (480, 480)  # 使用默认输入尺寸
    
    # 检查图像尺寸是否有效
    if img_size[0] <= 0 or img_size[1] <= 0:
        raise ValueError(f"无效的图像尺寸: {img_size}")
    
    mat, mat_inv = getTransformMat(img_size, input_size, True)
    
    # 图像变换
    img_transformed = cv2.warpAffine(
        img_cv,
        mat,
        input_size,
        flags=cv2.INTER_CUBIC,
        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    )
    
    # 转换为RGB并归一化
    img_rgb = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)
    img_tensor = convert(img_rgb)
    
    # 文本tokenize
    text = tokenize(description, 17, True)  # 使用默认word_len=17
    
    # 移动到GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
    
    # 添加batch维度
    img_tensor = img_tensor.unsqueeze(0)
    
    # 模型推理
    with torch.no_grad():
        pred = model(img_tensor, text)
        
        # 确保预测结果尺寸与输入一致
        if pred.shape[-2:] != img_tensor.shape[-2:]:
            pred = F.interpolate(pred,
                                size=img_tensor.shape[-2:],
                                mode='bilinear',
                                align_corners=True).squeeze()
        
        # Sigmoid激活
        pred = torch.sigmoid(pred)
    
    # 后处理
    pred = pred.cpu().numpy()
    
    # 确保pred是2D图像
    if len(pred.shape) == 4:
        pred = pred.squeeze()  # 移除batch和channel维度
    elif len(pred.shape) == 3:
        pred = pred.squeeze()  # 移除channel维度
    
    pred = cv2.warpAffine(pred, mat_inv, (img_size[1], img_size[0]),
                          flags=cv2.INTER_CUBIC,
                          borderValue=0.)
    
    # 二值化
    pred_binary = np.array(pred > 0.35, dtype=np.uint8)
    
    # 创建结果字典
    result = {
        'mask': pred_binary,  # 二值化掩码
        'probability': pred,  # 概率图
        'original_image': ori_img,  # 原始图像
        'description': description  # 输入描述
    }
    
    return result


def load_model(config_path: str = 'config/general_inference.yaml') -> Any:
    """
    加载预训练模型
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        加载好的模型
    """
    # 加载配置
    cfg = config.load_cfg_from_cfg_file(config_path)
    cfg.input_size = (cfg.input_size, cfg.input_size)
    
    # 构建模型
    model, _ = build_segmenter(cfg)
    model = torch.nn.DataParallel(model)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 加载预训练权重
    if os.path.isfile(cfg.model_dir):
        logger.info(f"=> loading checkpoint '{cfg.model_dir}'")
        checkpoint = torch.load(cfg.model_dir, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info(f"=> loaded checkpoint '{cfg.model_dir}'")
    else:
        raise ValueError(f"预训练模型文件未找到: {cfg.model_dir}")
    
    return model


def save_visualization(result: Dict, output_path: str):
    """
    保存可视化结果
    
    Args:
        result: forward函数返回的结果字典
        output_path: 输出文件路径
    """
    mask = result['mask']
    
    # 保存掩码图像
    mask_img = np.array(mask * 255, dtype=np.uint8)
    cv2.imwrite(output_path, mask_img)
    
    logger.info(f"结果已保存到: {output_path}")


def main():
    """
    主函数：处理命令行参数并执行推理
    """
    parser = argparse.ArgumentParser(description='UniLSeg Demo - 单张图片分割')
    parser.add_argument('--input_image_path', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--prompt', type=str, required=True,
                        help='目标对象的描述文本')
    parser.add_argument('--output_image_path', type=str, required=True,
                        help='输出可视化结果路径')
    parser.add_argument('--config', type=str, default='config/general_inference.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_image_path):
        raise FileNotFoundError(f"输入图像文件不存在: {args.input_image_path}")
    
    # 加载模型
    logger.info("正在加载模型...")
    model = load_model(args.config)
    
    # 加载图像
    logger.info(f"正在处理图像: {args.input_image_path}")
    image = Image.open(args.input_image_path).convert('RGB')
    
    # 执行推理
    logger.info(f"正在处理描述: {args.prompt}")
    result = forward(model, image, args.prompt)
    
    # 保存结果
    save_visualization(result, args.output_image_path)
    
    logger.info("处理完成！")


if __name__ == '__main__':
    main() 