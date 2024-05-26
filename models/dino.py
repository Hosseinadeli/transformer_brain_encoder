# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DINO Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Callable

from utils.utils import NestedTensor

from .position_encoding import build_position_encoding
    
    
class dino_model_with_hooks(nn.Module):

    def __init__(self, enc_output_layer, return_interm_layers= False):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)  #self.hook_fn_forward_qkv())
        
        self.return_interm_layers = return_interm_layers

    def hook_fn_forward_qkv(self, module, input, output) -> Callable:
#         def fn(_, __, output):
        self.qkv_feats['qkv_feats'] = output
            
            
    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        
        #print(xs.shape)
        h, w = int(xs.shape[2]/14), int(xs.shape[3]/14)
        
#         self.qkv_feats = []    
#         qkv_feats = []
            
#         self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(lambda self, input, output: qkv_feats.append(output))
        
        xs = self.backbone.get_intermediate_layers(xs)[0]

        feats = self.qkv_feats['qkv_feats']
        # Dimensions
        nh = 12 #Number of heads
        
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        
        xs = q[:,1:,:]

        xs = {'layer_top':xs}
#         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0],h,w,self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    

class dino_model(nn.Module):

    def __init__(self, enc_output_layer, return_interm_layers= False):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.enc_output_layer = enc_output_layer
        self.return_interm_layers = return_interm_layers
        

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        
        patch_size = 14
    
        w_p = int(xs.shape[2] / patch_size)
        h_p = int(xs.shape[3] / patch_size)
        
        xs = self.backbone.get_intermediate_layers(xs, n=12) #[0]

        if self.return_interm_layers:
            xs = {'0':xs[0], '1':xs[1], '2':xs[2], '3':xs[3], '4':xs[4], '5':xs[5], '6':xs[6], '7':xs[7], '8':xs[8], '9':xs[9], '10':xs[10], '11':xs[11]}
        else:
            xs = {'layer_top':xs[self.enc_output_layer]}

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], w_p,h_p,self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
