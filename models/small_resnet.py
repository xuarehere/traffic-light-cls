'''
Author: xuarehere xuarehere@foxmail.com
Date: 2025-04-09 15:26:19
LastEditTime: 2025-04-09 15:26:22
LastEditors: xuarehere xuarehere@foxmail.com
Description: 
FilePath: /traffic-light-cls/modesl/small_resnet.py

'''
# models/small_resnet.py
import torch.nn as nn
import torch
from torchvision.models.resnet import BasicBlock
from torchvision.models import resnet18


class SmallResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SmallResNet, self).__init__()
        
        self.inplanes = 32  # 原本是 64

        # stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 主干网络，逐层减半
        self.layer1 = self._make_layer(BasicBlock, 32, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 如果输入维度不等于输出维度或 stride 不为1，需要降采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # [B, 32, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 32, H/4, W/4]

        x = self.layer1(x)  # [B, 32, H/4, W/4]
        x = self.layer2(x)  # [B, 64, H/8, W/8]
        x = self.layer3(x)  # [B, 128, H/16, W/16]
        x = self.layer4(x)  # [B, 256, H/32, W/32]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def load_pretrained_weights(small_model):
    # 加载 PyTorch 官方预训练的 ResNet18 模型
    pretrained = resnet18(pretrained=True)

    pretrained_dict = pretrained.state_dict()
    model_dict = small_model.state_dict()

    new_state_dict = {}

    for k in model_dict.keys():
        if k in pretrained_dict:
            # 如果 shape 匹配则直接加载
            if model_dict[k].shape == pretrained_dict[k].shape:
                new_state_dict[k] = pretrained_dict[k]
            # 特别处理conv层，裁剪通道
            elif 'conv' in k and len(model_dict[k].shape) == 4:
                # e.g. conv.weight: [out, in, k, k]
                pre_weight = pretrained_dict[k]
                new_shape = model_dict[k].shape
                new_weight = pre_weight[:new_shape[0], :new_shape[1], :, :]
                new_state_dict[k] = new_weight
            # 特别处理 BN 层
            elif 'bn' in k:
                pre_weight = pretrained_dict[k]
                new_weight = pre_weight[:model_dict[k].shape[0]]
                new_state_dict[k] = new_weight
            else:
                # 其他不匹配跳过
                pass
        else:
            # 模型中新增参数项（如fc）跳过
            pass

    # 更新模型参数
    model_dict.update(new_state_dict)
    small_model.load_state_dict(model_dict)

    print(f"[✓] Loaded {len(new_state_dict)} pretrained layers into small model.")