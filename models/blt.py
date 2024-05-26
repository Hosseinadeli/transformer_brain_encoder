import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import torch.utils.model_zoo

class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """
    def forward(self, x):
        return x

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class blt(nn.Module):

    def __init__(self, conn_matrix, num_classes, layer_channels, out_shape, times=5):
        super().__init__()
        self.times = times
        self.num_classes = num_classes
        self.connections = {}
        self.non_lins = {}
        self.layer_channels = layer_channels 
        self.conn_matrix = conn_matrix
        num_layers = len(conn_matrix)

        if out_shape['0'] == 56:
            self.conv_input = nn.Conv2d(self.layer_channels['inp'], self.layer_channels['0'], 
                                        kernel_size=7, stride=4, padding=3) # 5/2  7/4
        elif out_shape['0'] == 112:
            self.conv_input = nn.Conv2d(self.layer_channels['inp'], self.layer_channels['0'], 
                                        kernel_size=5, stride=2, padding=2) 

        self.non_lin_input =  nn.ReLU(inplace=True)
        self.norm_input = nn.GroupNorm(32, self.layer_channels['0'])

        # define all the connections between the layers
        for i in range(num_layers):
            setattr(self, f'non_lin_{i}', nn.ReLU(inplace=True))
            setattr(self, f'norm_{i}', nn.GroupNorm(32, self.layer_channels[f'{i}']))
            setattr(self, f'output_{i}', Identity())
            for j in range(num_layers):
                if not conn_matrix[i,j]: continue

                setattr(self, f'norm_{i}_{j}', nn.GroupNorm(32, self.layer_channels[f'{j}']))
                setattr(self, f'non_lin_{i}_{j}', nn.ReLU(inplace=True))

                # bottom-up or lateral connection
                if i <= j:
                    conn_len = j - i

                    shape_factor = out_shape[f'{i}'] // out_shape[f'{j}'] 
                    
                    if shape_factor == 1:
                        cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                    elif shape_factor == 2:
                        cnn_kwargs = dict(kernel_size=3, stride=2, padding=1)
                    elif shape_factor == 4:
                        cnn_kwargs = dict(kernel_size=5, stride=4, padding=2)
                    elif shape_factor == 8:
                        cnn_kwargs = dict(kernel_size=9, stride=8, padding=4)
                    elif shape_factor == 16:
                        cnn_kwargs = dict(kernel_size=17, stride=16, padding=8)

                    conn =  nn.Conv2d(self.layer_channels[f'{i}'], 
                                      self.layer_channels[f'{j}'], 
                                      **cnn_kwargs)
                    
                    setattr(self, f'conv_{i}_{j}', conn)

                # top-down connections
                elif i > j:
                    conn_len = i - j
                    shape_factor = out_shape[f'{j}'] // out_shape[f'{i}'] 

                    if shape_factor == 1:
                        #cnn_kwargs = dict(kernel_size=3, stride=1, padding=1, output_padding=1)
                        cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                        conn =  nn.Conv2d(self.layer_channels[f'{i}'], 
                                      self.layer_channels[f'{j}'], 
                                      **cnn_kwargs)
                    else:
                        if shape_factor == 2:
                            cnn_kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
                        elif shape_factor == 4:
                            cnn_kwargs = dict(kernel_size=5, stride=4, padding=2, output_padding=3)
                        elif shape_factor == 8:
                            cnn_kwargs = dict(kernel_size=9, stride=8, padding=4, output_padding=7)
                        elif shape_factor == 16:
                            cnn_kwargs = dict(kernel_size=17, stride=16, padding=8, output_padding=15)

                        conn = nn.ConvTranspose2d(self.layer_channels[f'{i}'],
                                                self.layer_channels[f'{j}'],
                                                **cnn_kwargs)

                    setattr(self, f'conv_{i}_{j}', conn)
                    

        self.read_out = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, self.num_classes))
        ]))


    def forward(self, inp):
        outputs = {} #'inp': inp
        states = {}
        blocks = list(self.layer_channels.keys()) #['inp', '0', '1', '2', '3']

        inp = self.conv_input(inp)
        inp = self.norm_input(inp)
        inp = self.non_lin_input(inp)
        outputs[blocks[1]] = getattr(self, f'output_{blocks[1]}')(inp)
        for block in blocks[2:]:
            outputs[block] = None

        for t in range(1, self.times):
            new_outputs = {blocks[1]: outputs[blocks[1]]}  # {'0': inp}
            for block in blocks[1:]:

                output_prev_step = outputs[block]
                
                conn_input = 0
                in_blocks =  self.conn_matrix[:,int(block)] 
                for i in np.where(in_blocks)[0]: 
                    if outputs[f'{i}'] is not None:
                        input = getattr(self, f'conv_{i}_{block}')(outputs[f'{i}'])
                        input = getattr(self, f'norm_{i}_{block}')(input)
                        input = getattr(self, f'non_lin_{i}_{block}')(input)
                        conn_input += input
                

                if output_prev_step is not None:
                    new_output = output_prev_step + conn_input
                elif conn_input is not 0:
                    new_output = conn_input
                else:
                    new_output = None

                # apply relu here   
                if new_output is not None:
                    new_output = getattr(self, f'norm_{block}')(new_output)
                    new_output = getattr(self, f'non_lin_{block}')(new_output)
                    new_output = getattr(self, f'output_{block}')(new_output)

                new_outputs[block] = new_output
                
            outputs = new_outputs

        out = outputs[blocks[-1]]
        return self.read_out(out)


def get_blt_model(model_name, pretrained=False, map_location=None, **kwargs):

    num_layers = kwargs['num_layers']
    img_channels = kwargs['in_channels']
    times = kwargs['times']
    num_classes = kwargs['num_classes']
    
    if num_layers == 4:
        layer_channels  = {'inp':img_channels, '0':64, '1':128, '2':256, '3':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7}
    elif num_layers == 5:
        layer_channels = {'inp':img_channels, '0':64, '1':128, '2':128, '3':256, '4':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7, '4':7}
    elif num_layers == 6:
        # layer_channels = {'inp':img_channels, '0':64, '1':64, '2':128, '3':128, '4':256, '5':512}
        # out_shape  = {'0':56, '1':28, '2':14, '3':14, '4':7, '5':7}
        layer_channels = {'inp':img_channels, '0':64, '1':128, '2':128, '3':128, '4':256, '5':512}
        out_shape  = {'0':112, '1':56, '2':28, '3':14, '4':7, '5':7}

    num_layers = len(list(layer_channels.keys())) - 1                         
    conn_matrix = np.zeros((num_layers, num_layers))

    shift = [-1]  # this corresponds to bottom up connections -- always present
    if 'l' in model_name: shift = shift + [0]
    if 'b2' in model_name: shift = shift + [-2]
    if 'b3' in model_name: shift = shift + [-2, -3]
    if 't' in model_name: shift = shift + [1] 
    if 't2' in model_name: shift = shift + [1, 2]
    if 't3' in model_name: shift = shift + [2, 3]
    for i in range(num_layers):
        for j in range(num_layers):
            for s in shift:
                # just add other connections for the last 4 layers
                if (s != -1) and ((i<(num_layers-4)) or (j<(num_layers-4))): continue 
                if i == (j+s):
                    conn_matrix[i, j] = 1

    model = blt(conn_matrix, num_classes, layer_channels, out_shape, times)
    return model
