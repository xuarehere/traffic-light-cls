'''
Author: xuarehere xuarehere@foxmail.com
Date: 2025-04-21 19:13:23
LastEditTime: 2025-04-21 19:32:58
LastEditors: xuarehere xuarehere@foxmail.com
Description: 
FilePath: /traffic-light-cls/check-resnet/resnet34.py

'''
import time
import torch
import torchvision.models as models
import tensorrt

def benchmark(model, input_tensor, num_runs=100, warm_up=10):
    """对 model(input_tensor) 运行 num_runs 次，返回平均推理时间（单位：秒）。"""
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(warm_up):
            model(input_tensor)
        torch.cuda.synchronize()
        # 计时
        start = time.time()
        for _ in range(num_runs):
            model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    return elapsed / num_runs

def main():
    device = torch.device("cuda")

    # 1) 加载 PyTorch ResNet34 FP32 模型
    pt_model = models.resnet34(pretrained=True).to(device).eval()
    # 随机输入
    input_fp32 = torch.randn(1, 3, 224, 224, device=device)

    # 2) 测试 PyTorch FP32 推理速度
    pt_avg_time = benchmark(pt_model, input_fp32)
    print(f"PyTorch FP32 推理: {pt_avg_time * 1000:.2f} ms")

    # 3) 转换为 TensorRT FP16
    trt_model = tensorrt.compile(
        pt_model,
        inputs=[
            tensorrt.Input(
                min_shape=input_fp32.shape,
                opt_shape=input_fp32.shape,
                max_shape=input_fp32.shape,
                dtype=torch.float16
            )
        ],
        enabled_precisions={torch.float16},      # FP16
        truncate_long_and_double=True            # 避免 long/double 类型不支持
    )

    # 将输入也转为半精度
    input_fp16 = input_fp32.half()

    # 4) 测试 TensorRT FP16 推理速度
    trt_avg_time = benchmark(trt_model, input_fp16)
    print(f"TensorRT  FP16 推理: {trt_avg_time * 1000:.2f} ms")

    # 5) 加速比
    speedup = pt_avg_time / trt_avg_time
    print(f"加速比: {speedup:.2f}x")

if __name__ == "__main__":
    main()
