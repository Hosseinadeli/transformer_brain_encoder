
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math

from utils.utils import NestedTensor


from .position_encoding import build_position_encoding
from typing import Dict, Iterable, Callable

from models.resnet import resnet_model
from models.dino import dino_model, dino_model_with_hooks
from models.clip import clip_model
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):

    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.return_interm

    if 'cls' in args.backbone_arch:
        return_cls = True
    else:
        return_cls = False
        # if args.backbone_arch == 'dinov2_cls':
        #     backbone = dino_model(-1*args.enc_output_layer, return_interm_layers)
        #     num_channels = backbone.num_channels
        # elif args.backbone_arch == 'dinov2_q_cls':
        #     backbone = dino_model_with_hooks(-1*args.enc_output_layer, return_interm_layers)
        #     num_channels = backbone.num_channels
        # elif args.backbone_arch == 'clip_cls':
        #     backbone = clip_model(-1*args.enc_output_layer, return_interm_layers)
        #     num_channels = backbone.num_channels

    #else:
    if 'resnet' in args.backbone_arch: 
        backbone = resnet_model(args.backbone_arch, train_backbone, return_interm_layers, args.dilation)
        num_channels = backbone.num_channels
    elif 'dinov2_q' in args.backbone_arch:
        backbone = dino_model_with_hooks(-1*args.enc_output_layer, return_interm_layers, return_cls)
        num_channels = backbone.num_channels
    elif 'dinov2' in args.backbone_arch:
        backbone = dino_model(-1*args.enc_output_layer, return_interm_layers, return_cls)
        num_channels = backbone.num_channels
    elif 'clip' in args.backbone_arch:
        backbone = clip_model(-1*args.enc_output_layer, return_interm_layers, return_cls)
        num_channels = backbone.num_channels
            

    if return_cls == True:
        model = backbone
    else:
        position_embedding = build_position_encoding(args.position_embedding, args.hidden_dim // 2)
        model = Joiner(backbone, position_embedding)
        
        
    model.num_channels = num_channels
    
    return model
