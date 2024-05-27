
import numpy as np


from models.dino import dino_model, dino_model_with_hooks


def get_transformer_activations(model, ims, enc_layers, dec_layers):

    # use lists to store the outputs via up-values
    enc_output, enc_attn_weights, dec_output, dec_attn_weights = [], [], [], []

    hooks = []
    # hooks = [
    #     model.input_proj.register_forward_hook(
    #         lambda self, input, output: inp_features.append(output)
    #     ),
    # ]

    # encoder tokens
    for i in range(enc_layers):
        hooks.append(model.transformer.encoder.layers[-i].register_forward_hook(
                lambda self, input, output: enc_output.append(output)))
    # encoder attention
    for i in range(enc_layers):
        hooks.append(model.transformer.encoder.layers[-i].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output)))
    #decoder tokens
    for i in range(dec_layers):
        hooks.append(model.transformer.decoder.layers[-i].register_forward_hook(
                lambda self, input, output: dec_output.append(output[1])))
    #decoder attention 
    for i in range(dec_layers):
        hooks.append(model.transformer.decoder.layers[-i].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])))

    # propagate through the model
    outputs = model(ims)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    
    enc_output = enc_output
    enc_attn_weights = enc_attn_weights
    dec_output = dec_output
    dec_attn_weights = dec_attn_weights

    return enc_output, enc_attn_weights, dec_output, dec_attn_weights








    

