import torch
import yaml
from torch import nn
import sys
from backbone import EfficientDetBackbone
import numpy as np

sys.path.insert(0, "./Yet-Another-EfficientDet-Pytorch-Convert-ONNX-TVM")

weights_path = r'/content/efficientdet-d1_0610.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

model = EfficientDetBackbone(num_classes=48, compound_coef=1, onnx_export=True,
                                 ratios=eval(anchors_ratios), scales=eval(anchors_scales)).to(device)

model.backbone_net.model.set_swish(memory_efficient=False)

dummy_input = torch.randn((1,3,640,640), dtype=torch.float32).to(device)

model.load_state_dict(torch.load(weights_path,map_location=device),strict=False)

# opset_version can be changed to 10 or other number, based on your need
torch.onnx.export(model, dummy_input,
                  './efficientdet-d1-0610.onnx',
                  verbose=False,
                  input_names=['data'],
                  opset_version=11)
