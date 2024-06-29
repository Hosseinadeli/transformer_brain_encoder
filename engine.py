import math
import sys
from typing import Iterable
import torch
from tqdm import tqdm

from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

import utils.utils as utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    print_freq = 100
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=print_freq, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_labels', utils.SmoothedValue(window_size=print_freq))  #, fmt='{value:.2f}'
    header = 'Epoch: [{}]'.format(epoch)
    

    for imgs, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        if isinstance(imgs, (list, torch.Tensor)):
            imgs = tuple(imgs.cuda())
            imgs = nested_tensor_from_tensor_list(imgs)
    
        # TODO there may be a better way to do this 
        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        outputs = model(imgs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)  #, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        metric_logger.update(loss_labels=loss_value) #loss_dict_reduced['loss_recon']
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, args, lh_challenge_rois=None, rh_challenge_rois=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_labels', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    lh_f_pred_val = []
    rh_f_pred_val = []
    
    lh_fmri_val = []
    rh_fmri_val = []

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        
        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)

        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        outputs = model(samples)
    
        lh_f = targets[0]['lh_f']
        rh_f = targets[0]['rh_f']

        lh_fmri_val.append(lh_f.cpu().numpy())
        rh_fmri_val.append(rh_f.cpu().numpy())

        lh_f_pred = outputs['lh_f_pred']
        rh_f_pred = outputs['rh_f_pred']
        
        if (args.readout_res != 'hemis') and (args.readout_res != 'voxels'):
            lh_f_pred = outputs['lh_f_pred'][:,:,:args.roi_nums]
            rh_f_pred = outputs['rh_f_pred'][:,:,:args.roi_nums]
        
            lh_challenge_rois_b = torch.tile(lh_challenge_rois[:,:,None], (1,1,lh_f_pred.shape[0])).permute(2,1,0)
            rh_challenge_rois_b = torch.tile(rh_challenge_rois[:,:,None], (1,1,rh_f_pred.shape[0])).permute(2,1,0)

            lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
            rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)
        
        lh_f_pred_val.append(lh_f_pred.cpu().numpy())
        rh_f_pred_val.append(rh_f_pred.cpu().numpy())
    
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)

                             #loss_value = losses_reduced_scaled.item()
                
        metric_logger.update(loss_labels=loss_dict_reduced['loss_labels'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return np.concatenate(lh_f_pred_val, axis=0), np.concatenate(rh_f_pred_val, axis=0), np.concatenate(lh_fmri_val, axis=0), np.concatenate(rh_fmri_val, axis=0), loss_dict_reduced['loss_labels']



@torch.no_grad()
def test(model, criterion, data_loader, args, lh_challenge_rois, rh_challenge_rois):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_labels', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    
    lh_f_pred_all = []
    rh_f_pred_all = []
    
    for i,samples in tqdm(enumerate(data_loader), total=len(data_loader)):

        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)

        outputs = model(samples)
        
        lh_f_pred = outputs['lh_f_pred']
        rh_f_pred = outputs['rh_f_pred']
        
        if (args.readout_res != 'hemis') and (args.readout_res != 'voxels'):
            lh_f_pred = outputs['lh_f_pred'][:,:,:args.roi_nums]
            rh_f_pred = outputs['rh_f_pred'][:,:,:args.roi_nums]
        
            lh_challenge_rois_b = torch.tile(lh_challenge_rois[:,:,None], (1,1,lh_f_pred.shape[0])).permute(2,1,0)
            rh_challenge_rois_b = torch.tile(rh_challenge_rois[:,:,None], (1,1,rh_f_pred.shape[0])).permute(2,1,0)

            lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
            rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)
        
        lh_f_pred_all.append(lh_f_pred.cpu().numpy())
        rh_f_pred_all.append(rh_f_pred.cpu().numpy())
        
    return np.vstack(lh_f_pred_all), np.vstack(rh_f_pred_all)