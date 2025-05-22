import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.custom_transformer import build_custom_transformer

class brain_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lr_backbone = args.lr_backbone

        self.backbone_arch = args.backbone_arch
        self.return_interm = args.return_interm
        self.encoder_arch = args.encoder_arch

        self.lh_vs = args.lh_vs
        self.rh_vs = args.rh_vs

        ### backbone_arch for feature exraction
        self.backbone_model = build_backbone(args)

        # number of brain areas
        self.num_queries = args.num_queries

        #TODO hard  coding the map size for now but fix it
        self.map_size = 31

        ### Brain encoding model
        if 'transformer' in args.encoder_arch:
            if args.encoder_arch == 'transformer':
                self.transformer = build_transformer(args)

            elif self.encoder_arch == 'custom_transformer':
                self.transformer = build_custom_transformer(args)


            self.hidden_dim = self.transformer.d_model
            self.linear_feature_dim  = self.hidden_dim
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

            if ('resnet' in self.backbone_arch):
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)
        
        elif self.encoder_arch == 'spatial_feature':

            if 'clip' in self.backbone_arch:
                self.map_size = 16

            self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
            self.linear_feature_dim = self.backbone_model.num_channels

            self.downsize = False
            if self.downsize: 
                self.hidden_dim = 256
                if 'resnet' in self.backbone_arch:
                    stride=1
                    self.map_size = 11
                elif 'clip' in self.backbone_arch:
                    stride=1
                    self.map_size = 8
                else:
                    stride=3
                    self.map_size = 11

                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                    
                # for each roi, learn a spatial map
                self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
                self.linear_feature_dim = self.hidden_dim

        elif self.encoder_arch == 'linear':
            #TODO hard  coding the map size and hidden dimention for now but fix it
            # using conv to make the input smaller for linear layer
            
            if 'resnet' in self.backbone_arch:
                self.hidden_dim = 256
                stride=1
                self.map_size = 11
            elif 'clip' in self.backbone_arch:
                self.hidden_dim = 256
                stride=2
                self.map_size = 8
            else:
                self.hidden_dim = 256
                stride=3
                self.map_size = 11

            #if 'dino' in self.backbone_arch:
            self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                
            # if ('resnet' in self.backbone_arch):
            #     self.input_proj = nn.AdaptiveAvgPool2d(1)
            if 'cls' in self.backbone_arch:
                self.hidden_dim = 768
                self.linear_feature_dim  = self.hidden_dim
            else:
                self.linear_feature_dim = self.hidden_dim*self.map_size*self.map_size

        #what is the readout resolution - hemispheres, rois, voxels
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



        if 'cls' in self.backbone_arch:
            with torch.no_grad():
                input_proj_src = self.backbone_model(samples)

        else:
            if self.lr_backbone == 0:
                with torch.no_grad():
                    features, pos = self.backbone_model(samples)
            else:
                features, pos = self.backbone_model(samples)
        
            input_proj_src, mask = features[-1].decompose()
            assert mask is not None
            pos_embed = pos[-1]
            _,_,h,w = pos_embed.shape


        # print('input_proj_src.shape:', input_proj_src.shape)
        # print('mask.shape:', mask.shape)
        # print(mask)
        # print('pos_embed.shape:', pos_embed.shape)

        # pos_embed = torch.zeros_like(pos_embed).to(pos_embed.device)

        if self.encoder_arch == 'transformer':
            
        # if backbone is resnet, apply 1x1 conv to project the feature to the transformer dimension
            if 'resnet' in self.backbone_arch:
                input_proj_src = self.input_proj(input_proj_src)

            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'voxels':

                lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

            elif self.readout_res == 'hemis':
                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:
                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'custom_transformer':

            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'voxels':

                lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

            elif self.readout_res == 'hemis':
                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:
                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'spatial_feature':

            if self.downsize:
                input_proj_src = self.input_proj(input_proj_src)
            
            if self.readout_res == 'rois_all':
                # only for rois_all
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            elif self.readout_res == 'voxels':
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:self.lh_vs,:])
                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)

                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)


            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'linear':
            #if 'dino' in self.backbone_arch:
            if 'cls' not in self.backbone_arch: 
                input_proj_src = self.input_proj(input_proj_src)

            output_tokens = input_proj_src.flatten(1)
            lh_f_pred = self.lh_embed(output_tokens)
            rh_f_pred = self.rh_embed(output_tokens)

            l2_reg = torch.tensor(0.).cuda()
            for param in self.lh_embed.parameters():
                l2_reg += torch.norm(param)

            for param in self.rh_embed.parameters():
                l2_reg += torch.norm(param)  

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg}
        

        return out
