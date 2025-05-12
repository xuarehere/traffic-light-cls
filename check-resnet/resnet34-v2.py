#!/usr/bin/env python3
"""
ResNet-34 PyTorch → TensorRT 性能对比（FP32 & FP16，无 cudart 依赖）

仅使用以下包：
import os
import time
import ctypes
import numpy as np
import torch
from torchvision import models
import tensorrt as trt
from torch.utils.data import Dataset

功能：
 - 加载 PyTorch 自带 ResNet-34
 - PyTorch FP32 & FP16 推理
 - 导出 ONNX，并构建 TensorRT 引擎（FP32 & FP16）
 - 使用 torch 分配 CUDA 缓冲，避免直接调用 cudart
 - 测量并打印各模式平均推理延迟（ms）
"""
import os
import time
import ctypes  # 保留但不调用 cudart
import numpy as np
import torch
from torchvision import models
import tensorrt as trt
from torch.utils.data import Dataset

# 检查 CUDA
if not torch.cuda.is_available():
    raise RuntimeError('CUDA 不可用，请检查环境。')
device = torch.device('cuda')

def export_onnx(model, onnx_path):
    model.to(device).eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'], opset_version=11
    )

class TRTBuilder:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file
        self.logger = trt.Logger(trt.Logger.WARNING)

    def build_engine(self, precision='fp32'):
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        with open(self.onnx_file, 'rb') as f:
            if not parser.parse(f.read()):
                raise RuntimeError('ONNX 解析失败')
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        return builder.build_engine(network, config)

# TensorRT 推理 & 基准（不使用 cudart）
def infer_trt(engine, precision_label, iterations=200, warmup=50):
    context = engine.create_execution_context()
    # 随机输入
    input_np = np.random.random((1, 3, 224, 224)).astype(np.float32)
    # 分配 GPU 缓冲
    d_input = torch.from_numpy(input_np).to(device)
    d_output = torch.empty_like(d_input)
    input_ptr = int(d_input.data_ptr())
    output_ptr = int(d_output.data_ptr())
    # 预热
    for _ in range(warmup):
        context.execute_v2([input_ptr, output_ptr])
    torch.cuda.synchronize()
    # 测时
    start = time.time()
    for _ in range(iterations):
        context.execute_v2([input_ptr, output_ptr])
    torch.cuda.synchronize()
    avg_ms = (time.time() - start) * 1000 / iterations
    print(f"TensorRT {precision_label}: {avg_ms:.2f} ms")
    return avg_ms

# PyTorch 推理 & 基准
from torch.cuda.amp import autocast

def infer_torch(model, input_tensor, precision_label, use_fp16=False, iterations=200, warmup=50):
    model.eval()
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            if use_fp16:
                with autocast(): model(input_tensor)
                # model(input_tensor)
            else:
                model(input_tensor)
    torch.cuda.synchronize()
    # 测时
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iterations):
        starter.record()
        with torch.no_grad():
            if use_fp16:
                with autocast(): model(input_tensor)
            else:
                model(input_tensor)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    avg_ms = sum(times) / len(times)
    print(f"PyTorch {precision_label}: {avg_ms:.2f} ms")
    return avg_ms

if __name__ == '__main__':
    # 准备模型与输入
    model_fp32 = models.resnet50(pretrained=False).to(device)
    input_fp32 = torch.randn(1, 3, 224, 224, device=device)
    model_fp16 = models.resnet50(pretrained=False).to(device).half()
    input_fp16 = input_fp32.half()

    # 导出 ONNX
    onnx_file = 'resnet50.onnx'
    if not os.path.exists(onnx_file):
        export_onnx(model_fp32, onnx_file)

    # 构建 TensorRT 引擎
    builder = TRTBuilder(onnx_file)
    engine_fp32 = builder.build_engine('fp32')
    engine_fp16 = builder.build_engine('fp16')

    # 运行基准测试
    pt32 = infer_torch(model_fp32, input_fp32, 'FP32', use_fp16=False)
    pt16 = infer_torch(model_fp16, input_fp16, 'FP16', use_fp16=True)
    trt32 = infer_trt(engine_fp32, 'FP32')
    trt16 = infer_trt(engine_fp16, 'FP16')

    # 汇总结果
    print("--- 性能对比汇总 (ms) ---")
    print(f"PyTorch FP32: {pt32:.2f}")
    print(f"PyTorch FP16: {pt16:.2f}")
    print(f"TensorRT FP32: {trt32:.2f}")
    print(f"TensorRT FP16: {trt16:.2f}")

    """_summary_
    
    resnet 50
    --- 性能对比汇总 (ms) ---
    PyTorch FP32: 8.52
    PyTorch FP16: 10.52
    TensorRT FP32: 1.17
    TensorRT FP16: 0.57
    
    resnet 34
    --- 性能对比汇总 (ms) ---
    PyTorch FP32: 4.33
    PyTorch FP16: 5.57
    TensorRT FP32: 2.12
    TensorRT FP16: 1.07
    
    resnet 18
    --- 性能对比汇总 (ms) ---
    PyTorch FP32: 3.71
    PyTorch FP16: 8.20
    TensorRT FP32: 1.36
    TensorRT FP16: 0.58    
    """
    
    