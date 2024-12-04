# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_enc=False, return_intermediate_dec=False, enc_output_layer=-1):
        super().__init__()
        
        self.num_encoder_layers = num_encoder_layers

        if self.num_encoder_layers > 0:
            
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, return_intermediate=return_intermediate_enc)

            self.enc_output_layer = enc_output_layer
        
        self.num_decoder_layers = num_decoder_layers
        
        if self.num_decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)
            decoder_norm = None # nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
                                          
#         self.decoder = TransformerEncoder(encoder_layer, num_decoder_layers, encoder_norm)
        
        # self.norm1 = nn.LayerNorm(d_model)
        # self.linear1 = nn.Linear(d_model, d_model)

        self.memory_proj = nn.Conv2d(d_model, 64, kernel_size=1)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, masks, src_all=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        
        if self.num_encoder_layers > 0:
        
            memory_layers, sattn = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            memory = memory_layers[self.enc_output_layer]
            
        else:
            memory = src
            
            
        if masks: 
            memory = memory.permute(1, 2, 0)
            memory = torch.reshape(memory, (memory.shape[0], memory.shape[1], h, w))

            memory = self.memory_proj(memory)
            memory = torch.cat((memory, src_all), dim=1)

            memory = memory.flatten(2).permute(2, 0, 1)


        if self.num_decoder_layers > 0:
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
            
            return hs.transpose(1, 2) 
        
        return memory.permute(1, 2, 0).view(bs, c, h, w)

        
        # return memory_dec.permute(1, 2, 0).view(bs, c, h, w), memory.permute(1, 2, 0).view(bs, c, h, w)
        
            # sattn = sattn.reshape(h,w,h,w)
            
            # m = torch.nn.ZeroPad2d(8)

            # all_s_maps = []
            # for r_h in range(h):
                # for c_w in range(w):
                    # s_map = sattn[..., r_h, c_w]
                    # s_map = m(s_map)
                    
                    # s_map = s_map[r_h:r_h+16,c_w:c_w+16]
                    
                    # all_s_maps.append(s_map.reshape((1,256)))
                    
            # sattn = torch.stack(all_s_maps)
            
        # sattn = self.norm1(self.linear1(sattn))     
        
#             memory_dec, dec_sattn = self.decoder(memory, src_key_padding_mask=mask, pos=pos_embed)
#         return memory_dec.permute(1, 2, 0).view(bs, c, h, w), memory.permute(1, 2, 0).view(bs, c, h, w)
        
        # print(query_embed.shape)
        
        # print('memory min= {} max={}'.format(torch.min(memory), torch.max(memory)))
        # print('sattn min= {} max={}'.format(torch.min(sattn), torch.max(sattn)))
         
        # hs = self.decoder(tgt, memory+pos_embed, memory_key_padding_mask=mask, query_pos=query_embed)
        
        # hs = self.decoder(tgt+query_embed, memory, memory_key_padding_mask=mask,
                  # pos=pos_embed)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        intermediate_sattn = []
        for layer in self.layers:
            output, sattn = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            if self.return_intermediate:
                intermediate_sattn.append(sattn)
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
                
#         if self.norm is not None:
#             output = self.norm(output)
            
#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)
            
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_sattn)

        return output.unsqueeze(0), sattn.unsqueeze(0)  # encoder tokens and self-attention weights from the last layer


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, sattn = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, sattn

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, sattn = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        
        return src, sattn

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = nhead
        self.d_model = d_model
        self.dropout = dropout

        print('d_model:', d_model, 'nhead:', nhead)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            #TODO what is block_size? 
            block_size = 10
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, k, q, v):
        
        B, T_v, C = v.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T_q, C = q.size()

        self.n_head = 16

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = self.k_proj(k)
        q = self.q_proj(q)
        #v = self.v_proj(v)
        k = k.view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_v, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if 0: #self.flash:
            #print('we are using flash attention')
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        #print('y.shape:', y.shape) #[32, 16, 50, 48]
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.cross_attn = CrossAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
                    
        # for the first layer tgt is zero
        #tgt2 = self.norm1(tgt)
        
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)

        query=self.with_pos_embed(tgt, query_pos) #[50, 32, 768]
        query = torch.permute(query, [1,0,2]) #[32, 50, 768]
        key=self.with_pos_embed(memory, pos)  #[961, 32, 768]
        #key = memory
        key = torch.permute(key, [1,0,2]) #[32, 961, 768]
        value=torch.permute(memory, [1,0,2]) #[961, 32, 768]

        B, T, C = value.size()

        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))

        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ value # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) torch.Size([32, 50, 961])
        
        #tgt = y.transpose(0, 1) #.contiguous().view(B, T, C) 

        tgt = self.cross_attn(key, query, value).transpose(0, 1)

        #print('tgt.shape:', tgt.shape)  
        #print('query.shape:', query.shape, 'key.shape:', key.shape, 'value.shape:', value.shape)

        #return tgt.transpose(0, 1)

        # cross attention - torch.Size([50, 32, 768])
        # tgt = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=memory,
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        

        # tgt = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                     key=self.with_pos_embed(memory, pos),
        #                     value=memory, attn_mask=memory_mask,
        #                     key_padding_mask=memory_key_padding_mask)[0]
        

        #tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_custom_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_enc=True,
        return_intermediate_dec=False,
        enc_output_layer = args.enc_output_layer,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
