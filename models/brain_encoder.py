
import torch
from torch import nn
from collections import OrderedDict

from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_transformer

class brain_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lr_backbone = args.lr_backbone

        self.backbone_arch = args.backbone_arch
        self.return_interm = args.return_interm
        self.encoder_arch = args.encoder_arch

        ### feature extraction model
        if args.encoder_arch == 'transformer':
            self.transformer = build_transformer(args)

            self.num_queries = args.num_queries
            self.hidden_dim = self.transformer.d_model
            self.linear_feature_dim  = self.hidden_dim

            self.enc_layers = args.enc_layers
            self.dec_layers = args.dec_layers
            
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
            self.query_pos = torch.zeros_like(self.query_embed.weight)
            
        ### backbone_arch
        if self.backbone_arch:
            self.backbone_model = build_backbone(args)

            if ('resnet' in self.backbone_arch) and ('transformer' in self.encoder_arch):
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)

        else:
            if args.encoder_arch == 'transformer':
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)


        # readout layer to the neural data
        self.readout_res = args.readout_res
        
 
        self.lh_embed = nn.Sequential(
            nn.Linear(self.linear_feature_dim, args.lh_vs),
        )

        self.rh_embed = nn.Sequential(
            nn.Linear(self.linear_feature_dim, args.rh_vs),
        )
            

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        if self.backbone_arch:
            if self.lr_backbone == 0:
                with torch.no_grad():
                    features, pos = self.backbone_model(samples)
            else:
                features, pos = self.backbone_model(samples)
            
            src, mask = features[-1].decompose()
            assert mask is not None
            pos_embed = pos[-1]
            _,_,h,w = pos_embed.shape

            if 'dinov2' in self.backbone_arch:
                
                input_proj_src = src
                
            else:
                # only for the transformer readout the dim has to change 
                if self.encoder_arch == 'transformer':
                    input_proj_src = self.input_proj(src)
                else:
                    input_proj_src = src

        # If no backbone, then just feed the image tensors as inputs 
        else:
            input_proj_src = samples.tensors


        if self.encoder_arch == 'transformer':
    
            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'hemis':

                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:

                lh_f_pred = self.lh_embed(output_tokens[:,:8,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,8:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

        else:
            lh_f_pred = self.lh_embed(output_tokens)
            rh_f_pred = self.rh_embed(output_tokens)

        out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        return out
