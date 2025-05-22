

import open_clip
import torch
from utils.utils import NestedTensor
from typing import Dict, List, Callable
from torch import nn
import torch.nn.functional as F

class clip_model(nn.Module):

    def __init__(self, enc_output_layer, return_interm_layers= False, return_cls=False):
        super().__init__()   
        
        # Load model + transform
        self.backbone, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )

        # Ensure output tokens are returned (for patch-level features)
        self.backbone.visual.output_tokens = True
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.enc_output_layer = enc_output_layer
        self.return_interm_layers = return_interm_layers

        self.return_cls = return_cls
        

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        
        patch_size = 14
    
        w_p = int(xs.shape[2] / patch_size)
        h_p = int(xs.shape[3] / patch_size)

        cls_token, patch_tokens = self.backbone.visual(xs) 

        #print('cls_token.shape:', cls_token.shape, 'patch_tokens.shape:', patch_tokens.shape)

        # Project patch tokens from 1024 → 768 using projection layer
        proj = self.backbone.visual.proj  # shape: (1024, 768)
        patch_tokens_proj = patch_tokens @ proj  # (1, 256, 768)

        # Reshape CLS token to (1, 1, 768) for concat
        cls_token = cls_token.unsqueeze(1)  # (1, 1, 768)

        if self.return_cls:
            #out['cls_token'] = cls_token
            return cls_token

        # Concatenate CLS + patch tokens → (1, 257, 768)
        #full_tokens = torch.cat([cls_token, patch_tokens_proj], dim=1)
        #full_tokens = full_tokens.squeeze(0)  # (257, 768)

        # print('full_tokens.shape:', full_tokens.shape)
        # print('cls_token.shape:', cls_token.shape, 'patch_tokens_proj.shape:', patch_tokens_proj.shape)
        
        #xs = self.backbone.get_intermediate_layers(xs, n=12) #[0]

        # if self.return_interm_layers:
        #     xs = {'0':xs[0], '1':xs[1], '2':xs[2], '3':xs[3], '4':xs[4], '5':xs[5], '6':xs[6], '7':xs[7], '8':xs[8], '9':xs[9], '10':xs[10], '11':xs[11]}
        # else:
        xs = {'layer_top':patch_tokens_proj}
        #xs = {'cls_token':cls_token}

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0], w_p,h_p,self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        # 
        # print('out:', out)
        # print(out.keys())

        return out