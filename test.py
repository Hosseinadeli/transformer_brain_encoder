


import numpy as np
import torch


# TODO: Should I make this a standalone script? That can be called from the command line? 
# or just keep it as bunch of functions that can be imported and used in other scripts?

def get_activations_batch(model, ims, layer='output_3', sublayer='avgpool', time_step=0):
    """
    Kwargs:
        - model 
        - ims in format tensor[batch_size, channels, im_h, im_w]
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
    """
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(output)  #np.reshape(output, (len(output), -1))

    try:
        m = model.task_model
    except:
        m = model

    if 'output' in layer:      
        model_layer = getattr(m, layer)  #model.task_model.output_ #IT.output
    else:
        model_layer = getattr(getattr(m, layer), sublayer)

    hook = model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        _model_feats = []
        model(ims)
        if time_step:
            model_feats.append(_model_feats[time_step])
        else:
            model_feats.append(_model_feats)
        model_feats = np.concatenate(model_feats)

    hook.remove()    

    return model_feats