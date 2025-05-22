import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datasets.nsd_utils import roi_maps, roi_masks
from datasets.nsd import fetch_dataloaders
from scipy.stats import pearsonr as corr

from models.brain_encoder import brain_encoder
from engine import train_one_epoch, evaluate, test

import utils.utils as utils 
from pathlib import Path
import os

from PIL import Image
Image.warnings.simplefilter('ignore')

# np.random.seed(0)
# torch.manual_seed(0)

try:
    import wandb
    os.environ['WANDB_MODE'] = 'offline'
except ImportError as e:
    pass 


def get_args_parser():
    parser = argparse.ArgumentParser(description='NSD Training', add_help=False)

    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--output_path', default='./results/', type=str,
                        help='if not none, then store the model resuls')
    
    parser.add_argument('--save_model', default=False, type=int) 
    
    ## NSD params
    parser.add_argument('--subj', default=1, type=int) 
    parser.add_argument('--run', default=1, type=int)  
    parser.add_argument('--data_dir', default='../../../algonauts/algonauts_2023_challenge_data/', type=str)
    parser.add_argument('--parent_submission_dir', default='./algonauts_2023_challenge_submission/', type=str)
    
    parser.add_argument('--saved_feats', default=None, type=str) #'dinov2q'
    parser.add_argument('--saved_feats_dir', default='../../algonauts_image_features/', type=str) 
    
    parser.add_argument('--readout_res', choices=['voxels', 'rois_all', 'streams_inc', 'visuals', 'bodies', 'faces', 'places','words',
                                                  'hemis']
                        , default='streams_inc', type=str)   
    
    # the model for mapping from backbone image features to fMRI
    parser.add_argument('--encoder_arch', choices=['transformer', 'linear', 
                                                    'custom_transformer',
                                                    'spatial_feature'], 
                        default='transformer', type=str)
    
    parser.add_argument('--objective', choices=['NSD'],
                        default='classification', help='which model to train')
    
    parser.add_argument('--dataset', choices=['nsd_algo', 'nsd_gen'],
                        default='nsd_algo', help='which model to train')
    
    # Backbone
    parser.add_argument('--backbone_arch', choices=[None, 'dinov2', 'dinov2_q', 
                                                    'resnet18', 'resnet50',
                                                    'dinov2_cls', 'dinov2_q_cls',
                                                    'clip', 'clip_cls'],
                        default='dinov2_q', type=str,
                        help="Name of the backbone to use")  #resnet50 resnet18 dinov2
    
  
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--return_interm', default=False,
                        help="Train segmentation head if the flag is provided")

    # * Transformer
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer brain model")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer brain model")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")  #256  #868 (100+768) 
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=16, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=16, type=int,
                        help="Number of query slots")
    
    parser.add_argument('--pre_norm', default=1, type=int,
                        help="If 1, norm is applied before attention")
    
    parser.add_argument('--enc_output_layer', default=1, type=int,
                    help="Specify the encoder layer that provides the encoder output. default is the last layer")
    
    # training parameters
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading num_workers')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', default=.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay ')
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_backbone', default=0, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--evaluate', action='store_true', help='just evaluate')
    
    parser.add_argument('--wandb_p', default=None, type=str)
    parser.add_argument('--wandb_r', default=None, type=str)

    # dataset parameters
    parser.add_argument('--image_size', default=None, type=int, 
                        help='what size should the image be resized to?')
    parser.add_argument('--horizontal_flip', default=True,
                    help='wether to use horizontal flip augmentation')
    
    parser.add_argument('--img_channels', default=3, type=int,
                    help="what should the image channels be (not what it is)?") #gray scale 1 / color 3

    parser.add_argument('--distributed', default=False,
                        help='whether to use distributed training')

    return parser



class SetCriterion(nn.Module):
    def __init__(self, lh_challenge_rois, rh_challenge_rois):
        super().__init__()
        self.weight_dict = {'loss_labels': 1}

        self.readout_res = args.readout_res
        self.encoder_arch = args.encoder_arch
        self.backbone_arch = args.backbone_arch
        #roi_name_maps, lh_challenge_rois, rh_challenge_rois = roi_maps(args.data_dir)
        #self.roi_name_maps = roi_name_maps

        self.lh_challenge_rois = lh_challenge_rois
        self.rh_challenge_rois = rh_challenge_rois

        self.lh_rois_num = lh_challenge_rois.shape[0]
        self.rh_rois_num = rh_challenge_rois.shape[0]
        
        # self.lh_challenge_rois = torch.tensor(lh_challenge_rois).to(args.device)
        # self.rh_challenge_rois = torch.tensor(rh_challenge_rois).to(args.device)
        
        # args.lh_vs = len(lh_challenge_rois[args.rois_ind])
        # args.rh_vs = len(rh_challenge_rois[args.rois_ind])
        
        # self.rois_ind = args.rois_ind
        
        # self.lh_vs = args.lh_vs 
        # self.rh_v = args.rh_vs 

    def forward(self, outputs, targets):

        assert 'lh_f_pred' in outputs    
        assert 'rh_f_pred' in outputs 

        # TODO make target not a list 
        targets = targets[0]
        
        if (self.encoder_arch != 'linear') and (self.readout_res != 'hemis') and (self.readout_res != 'voxels'):

            lh_challenge_rois = torch.tile(self.lh_challenge_rois[:,:,None], (1,1,targets['lh_f'].shape[0])).permute(2,1,0)
            rh_challenge_rois = torch.tile(self.rh_challenge_rois[:,:,None], (1,1,targets['rh_f'].shape[0])).permute(2,1,0)
            
            outputs['lh_f_pred'] = torch.sum(torch.mul(lh_challenge_rois, outputs['lh_f_pred'][:,:,:self.lh_rois_num]), dim=2)
            outputs['rh_f_pred'] = torch.sum(torch.mul(rh_challenge_rois, outputs['rh_f_pred'][:,:,:self.rh_rois_num]), dim=2)

            if (self.readout_res != 'streams_inc') and (self.readout_res != 'rois_all'):

                outputs['lh_f_pred'] = (1*(lh_rois>0)) * outputs['lh_f_pred']
                outputs['rh_f_pred'] = (1*(rh_rois>0)) * outputs['rh_f_pred']

                targets['lh_f'] = (1*(lh_rois>0)) * targets['lh_f']
                targets['rh_f'] = (1*(rh_rois>0)) * targets['rh_f']
        
        loss_lh = nn.MSELoss()(outputs['lh_f_pred'], targets['lh_f'])
        loss_rh = nn.MSELoss()(outputs['rh_f_pred'], targets['rh_f'])
        #losses = {'loss_mse_fmri': loss_lh+loss_rh}

        loss = loss_lh+loss_rh

        # add a ridge penalty to the linear model
        if 'cls' not in self.backbone_arch:
            if self.encoder_arch == 'linear':
                loss = loss + 0.02* outputs['l2_reg']

        losses = {'loss_labels': loss}
        return losses
    

def main(rank, world_size, args):

    if args.distributed:
        args.rank = rank
        args.world_size = world_size
        utils.init_distributed_mode(args)
    else:
        args.gpu = 0

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    device = torch.device(args.device)

    args.val_perf = 0
    args.subj = format(args.subj, '02')
    args.data_dir = os.path.join(args.data_dir, 'subj'+ args.subj)
    
    if args.output_path:
        args.save_dir = args.output_path + f'nsd_test/{args.backbone_arch}_{args.encoder_arch}/subj_{args.subj}/{args.readout_res}/enc_{args.enc_output_layer}/run_{args.run}/'
        if (not os.path.exists(args.save_dir)) and (args.gpu == 0):
            os.makedirs(args.save_dir)

    if args.dataset == 'nsd_algo':

        roi_name_maps, lh_challenge_rois, rh_challenge_rois = roi_maps(args.data_dir)
        lh_challenge_rois_s, rh_challenge_rois_s, lh_roi_names, rh_roi_names, num_queries \
            = roi_masks(args.readout_res, roi_name_maps, lh_challenge_rois, rh_challenge_rois)

        lh_challenge_rois_s = lh_challenge_rois_s.to(args.device)
        rh_challenge_rois_s = rh_challenge_rois_s.to(args.device)

        print('roi_name_maps:', roi_name_maps)

        print('lh_challenge_rois:', len(lh_challenge_rois))
        print('lh_challenge_rois_s:', lh_challenge_rois_s.shape)

        args.num_queries = num_queries

        args.lh_vs = lh_challenge_rois_s.shape[1]
        args.rh_vs = rh_challenge_rois_s.shape[1]   

        #train_loader, val_loader = fetch_data_loaders(args)
        train_loader, val_loader = fetch_dataloaders(args, train='train')
        test_loader = fetch_dataloaders(args, train='test')

    elif args.dataset == 'nsd_gen':
        
        args.hemi = 'lh'
        args.data_dir = "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
        args.imgs_dir = "/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd"
        dataset = nsd_dataset_avg(args, split='train')
        train_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=32,
                            num_workers=4,
                            pin_memory=True,
                        )
    
        imgs, betas = next(iter(train_loader))
        print(betas['rh'].shape)

        
        neural_data_path = Path(
            "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
        )
        metadata = np.load(
            neural_data_path / f"metadata_sub-{int(args.subj):02}.npy", allow_pickle=True
        ).item()
        lh_roi_masks = metadata[f"lh_rois"] # returns roi, (num_voxels) where true if voxel is in roi
        rh_roi_masks = metadata[f"rh_rois"] # returns roi, (num_voxels) where true if voxel is in roi

        print(lh_roi_masks.keys())
        args.lh_vs = betas['lh'].shape[1]
        args.rh_vs = betas['rh'].shape[1]

    model = brain_encoder(args) #get_model(args)
    model = model.cuda() 
    num_parameters =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_parameters}")
    print(model)

    model_ddp = model
    if args.distributed:
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                              find_unused_parameters=True)
        
    criterion = SetCriterion(lh_challenge_rois_s, rh_challenge_rois_s)
    
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pretrained_dict = checkpoint['model']
        model.load_state_dict(pretrained_dict)
        
        args.best_val_acc = vars(checkpoint['args'])['val_perf'] #checkpoint['val_acc'] #or read it from the   
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            
            train_params = checkpoint['train_params']
            param_dicts = [ { "params" : [ p for n , p in model.named_parameters() if n in train_params ]}, ] 

            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                          weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
        
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    else:
        
        param_dicts = [ 
            { "params" : [ p for n , p in model.named_parameters() if p.requires_grad]}, ]  #n not in frozen_params and 
    
        train_params = [ n for n , p in model.named_parameters() if p.requires_grad ]  # n not in frozen_params and

        print('\ntrain_params', train_params)

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

        args.start_epoch = 0

    # only for one processs
    if args.gpu == 0: 
        if args.wandb_p:
            os.environ['WANDB_MODE'] = 'online'

            if args.wandb_r:
                wandb_r = args.wandb_r 
            else:
                wandb_r = args.encoder_arch 

            os.environ["WANDB__SERVICE_WAIT"] = "300"
            #        settings=wandb.Settings(_service_wait=300)
            wandb.init(
                # Set the project where this run will be logged
                project= args.wandb_p,   
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=wandb_r,  

                # Track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": f'{args.encoder_arch}',
                "epochs": args.epochs,
                })

        with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
            pprint.pprint(args.__dict__, f, sort_dicts=False)
        
        
        with open(os.path.join(args.save_dir, 'val_results.txt'), 'w') as f:
            f.write(f'validation results: \n') 

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
            
        train_stats = train_one_epoch(
            model_ddp, criterion, train_loader, optimizer, args.device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()


        # evaluate
        lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val, val_loss = evaluate(model, criterion, val_loader, args, lh_challenge_rois_s, rh_challenge_rois_s)

        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

        # Select the correlation results vertices of each ROI
        roi_names = []
        lh_roi_correlation = []
        rh_roi_correlation = []
        for r1 in range(len(lh_challenge_rois)):
            for r2 in roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                    lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                    rh_roi_correlation.append(rh_correlation[rh_roi_idx])
        roi_names.append('All vertices')
        lh_roi_correlation.append(lh_correlation)
        rh_roi_correlation.append(rh_correlation)


        # Create the plot
        lh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(lh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
            for r in range(len(lh_roi_correlation))]
        rh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(rh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
            for r in range(len(rh_roi_correlation))]

        val_perf = (lh_mean_roi_correlation[-1] + rh_mean_roi_correlation[-1]) / 2

        print('val_perf:', val_perf) 
        print('shape of rh_fmri_val_pred', rh_fmri_val_pred.shape)
        if (args.gpu == 0) and (args.wandb_p): 
            wandb_log = {"val_perf": val_perf}
            roi_clusters = {'visuals':np.arange(0,7), 'bodies': np.arange(7,11), 'faces':np.arange(11,16), 'places':np.arange(16,19),'words':np.arange(19,24)}
            for r in roi_clusters.keys():
                wandb_log[f'{r}'] = np.nanmean(np.array(lh_mean_roi_correlation)[roi_clusters[r]])
            wandb.log(wandb_log)

        if args.output_path:
            # update best validation acc and save best model to output dir
            if (val_perf > args.val_perf):  
                args.val_perf = val_perf                

                if args.gpu == 0: 
                    with open(os.path.join(args.save_dir, 'val_results.txt'), 'a') as f:
                            f.write(f'epoch {epoch}, val_perf: {val_perf} \n') 

                if args.save_model:
                    checkpoint_paths = [args.save_dir + '/checkpoint.pth']

                    model_state_dict = model.state_dict()
                    model_state_dict = {
                                k: v
                                for k, v in model_state_dict.items()
                                if "backbone_model" not in k
                            }
                    # print('checkpoint_path:',  checkpoint_paths)
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_state_dict,
                            # 'optimizer': optimizer.state_dict(),
    #                         'train_params' : train_params,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'val_perf': args.val_perf
                        }, checkpoint_path)

                np.save(args.save_dir+'lh_fmri_val_pred.npy', lh_fmri_val_pred)
                np.save(args.save_dir+'rh_fmri_val_pred.npy', rh_fmri_val_pred)

                np.save(args.save_dir+'lh_val_corr.npy', lh_correlation)
                np.save(args.save_dir+'rh_val_corr.npy', rh_correlation)

                lh_fmri_test_pred, rh_fmri_test_pred = test(model, criterion, test_loader, args, lh_challenge_rois_s, rh_challenge_rois_s)

                lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
                rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)

                np.save(args.save_dir+'/lh_pred_test.npy', lh_fmri_test_pred)
                np.save(args.save_dir+'/rh_pred_test.npy', rh_fmri_test_pred)

    if args.distributed:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # TODO: fix the shuffling issue before enabling distributed training
    # if args.distributed:
    #     args.world_size = torch.cuda.device_count()
    #     mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    # else:
    main(0, 1, args)