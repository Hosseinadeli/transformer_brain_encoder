

import torch
from models.activations import get_transformer_activations
from models.brain_encoder import brain_encoder
from datasets.nsd_utils import roi_maps, roi_masks
from engine import evaluate_batch
import numpy as np
import os
from scipy.special import softmax

class brain_encoder_wrapper():
    def __init__(self, subj=1, arch='dinov2_q_transformer', feature_name='dinov2_q_last',\
                 readout_res= 'rois_all', enc_output_layer=[1], runs=[1], results_dir=None, \
                  device=None, output_type='predictions'):
        
        self.readout_res = readout_res #'rois_all'
        self.enc_output_layer = enc_output_layer  # 1
        self.arch = arch # 'dinov2_q_transformer'
        self.subj = format(subj, '02')

        if results_dir is None:
            self.results_dir = '/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/results/'
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data/'
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
        # /engram/nklab/hossein/recurrent_models/transformer_brain_encoder/results/

        self.runs = runs

        roi_name_maps, lh_challenge_rois, rh_challenge_rois = roi_maps(self.data_dir)
        self.lh_challenge_rois, self.rh_challenge_rois, self.lh_roi_names, self.rh_roi_names, \
          numm_queries = roi_masks(self.readout_res, roi_name_maps, lh_challenge_rois, rh_challenge_rois)
        
        self.model = None
        # TODO up to how many models should/can I load? maybe put them on different GPUs?
        # if it is only a single model, load it here once
        if len(self.runs) == 1 and len(self.enc_output_layer) == 1:   
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.output_type = output_type
            model_path = f'{self.results_dir}/nsd_test/{self.arch}/subj_{self.subj}/{self.readout_res}/enc_{self.enc_output_layer[0]}/run_{self.runs[0]}/'
            self.model, self.args = self.load_model_path(model_path, self.device)  
        ## TODO what is the best way to load multiple models?
        else:
            total_runs = len(self.runs) * len(self.enc_output_layer)
            if readout_res == 'voxels':
                max_runs_per_gpu = 5
            else:
                max_runs_per_gpu = 20

            gpu_count = torch.cuda.device_count()
            gpu_ind = 0
            lh_correlation = []
            rh_correlation = []
            self.models = []
            run_on_gpu = 0
            for r in self.runs:
                for l in self.enc_output_layer:
                    
                    device = f'cuda:{gpu_ind}' if torch.cuda.is_available() else 'cpu'
                    print(f'Run {r} Backbone Layer {l} Device {device}')
                    model_path = f'{self.results_dir}/nsd_test/{self.arch}/subj_{self.subj}/{self.readout_res}/enc_{l}/run_{r}/'
                    model, _= self.load_model_path(model_path, device) 
                    self.models.append(model)

                    lh_correlation.append(np.load(model_path + 'lh_val_corr.npy'))
                    rh_correlation.append(np.load(model_path + 'rh_val_corr.npy'))

                    run_on_gpu += 1
                    if run_on_gpu == max_runs_per_gpu:
                        run_on_gpu = 0
                        gpu_ind += 1

                    if gpu_ind == gpu_count:
                        break
                if gpu_ind == gpu_count:
                    break
                
            
            lh_correlation = np.array(lh_correlation)
            lh_corr_sm = softmax(20*lh_correlation, axis=0)
            #lh_corr_sm = np.tile(np.expand_dims(lh_corr_sm,1), (1,lh_corr_sm.shape[1],1))
            self.lh_corr_sm = torch.tensor(lh_corr_sm)

            rh_correlation = np.array(rh_correlation)
            rh_corr_sm = softmax(20*rh_correlation, axis=0)
            #rh_corr_sm = np.tile(np.expand_dims(rh_corr_sm,1), (1,rh_corr_sm.shape[1],1))
            self.rh_corr_sm = torch.tensor(rh_corr_sm)

            self.output_type = output_type


    def load_model_path(self, model_path, device='cpu'):

        checkpoint = torch.load(model_path + 'checkpoint.pth', map_location='cpu')

        pretrained_dict = checkpoint['model']
        args = checkpoint['args']
        model = brain_encoder(args)
        model.load_state_dict(pretrained_dict)
        model.to(device)
        

        model.eval()

        try:
            model = model.module
        except:
            model = model
            
        model.device = device
        return model, args 

    def extract_transformer_features(self, model, imgs, enc_layers=0, dec_layers=1):

        model_features = {}
        try:
            model = model.module
        except:
            model = model

        outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights  = get_transformer_activations(model, imgs, enc_layers, dec_layers)

        return outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights
    
    # def combine_transformer_features(self, model, imgs, runs, enc_output_layers):
        
    #     for run in self.runs:
    #         for enc_output_layer in self.enc_output_layer:
                
    #     outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = \
    #       self.extract_transformer_features(self, model, imgs)
        

    def attention(self, images):

        #images = images.to(self.device)
        model_features = {}
        if self.model is not None:
            outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = \
                self.extract_transformer_features(self.model, images)
            
            #print('dec_attn_weights', len(dec_attn_weights), dec_attn_weights[0].shape)
            
            # model_features['outputs'] = outputs
            # model_features['enc_output'] = enc_output
            # model_features['enc_attn_weights'] = enc_attn_weights
            # model_features['dec_output'] = dec_output
            model_features['dec_attn_weights'] = dec_attn_weights

        else:
            dec_attn_weights_all = []
            for enc_output_layer in self.enc_output_layer:
                for run in self.runs:
                    print(f'Run {run}')
                    #subj = format(self.subj, '02')
                    model_path = f'{self.results_dir}/nsd_test/{self.arch}/subj_{self.subj}/{self.readout_res}/enc_{enc_output_layer}/run_{run}/'
                    model, _ = self.load_model_path(model_path)  

                    _, _, _, _, dec_attn_weights = \
                        self.extract_transformer_features(model, images.to(self.device))

                    dec_attn_weights_all.append(dec_attn_weights[0].detach().cpu().numpy()) 

                    del model


            model_features['dec_attn_weights'] = dec_attn_weights_all
    
        return model_features
    

    # def model_predictions(self, model, imgs):
    #     outputs = model(imgs)
    #     return outputs

    def forward(self, images):

        if self.model is not None:
            outputs_lh, outputs_rh = evaluate_batch(self.model, images.to(self.model.device), self.readout_res, self.lh_challenge_rois.to(self.model.device), self.rh_challenge_rois.to(self.model.device))
            return outputs_lh, outputs_rh
        else:
            outputs_lh = []
            outputs_rh = []
            for model in self.models:
                output_lh, output_rh = evaluate_batch(model, images.to(model.device), self.readout_res, self.lh_challenge_rois.to(model.device), self.rh_challenge_rois.to(model.device))
                outputs_lh.append(output_lh.to(self.device))
                outputs_rh.append(output_rh.to(self.device))

            outputs_lh = torch.stack(outputs_lh)
            outputs_rh = torch.stack(outputs_rh)

            
            lh_corr_sm = self.lh_corr_sm.unsqueeze(1).expand(-1, outputs_lh.size(1), -1).to(self.device) 
            lh_pred = (lh_corr_sm * outputs_lh).sum(0)  # Element-wise multiplication and summing along the first dimension
            
            print(lh_corr_sm[:,:,0:10])
            rh_corr_sm = self.rh_corr_sm.unsqueeze(1).expand(-1, outputs_rh.size(1), -1).to(self.device) 
            rh_pred = (rh_corr_sm * outputs_rh).sum(0)

            return lh_pred, rh_pred

                
        # outputs = np.array(outputs)
        # outputs = outputs.mean(0)   

        # dec_attn_weights_all = []
        # h, w = 31, 31
        # elif self.output_type == 'features':
        #   outputs, enc_output, enc_attn_weights, dec_output, dec_attn_weights = \
        #     self.extract_transformer_features(model, images.to(self.device))

        # dec_attn_weights_all.append(dec_attn_weights[0].reshape(-1,50,h, w).detach().cpu().numpy())

        #   dec_attn_weights_all = np.array(dec_attn_weights_all)
        #   dec_attn_weights = dec_attn_weights_all.mean(0)
      

# def simple_brain_encoder_wrapper():
    
#     class model_argObj:
#         def __init__(self, arch, feature_name, readout_res, enc_output_layer, learn_reg):

#             self.arch = arch
#             self.feature_name = feature_name
#             self.readout_res = readout_res
#             self.enc_output_layer = enc_output_layer
#             self.runs = np.arange(1,2)


#     qargs = []
#     qargs.append(['dinov2_q_transformer', 'dinov2_q_last', 'rois_all', [1], 0])
#     args = model_argObj(*qargs[0])

#     subj = 1
#     args.subj = format(subj, '02')
#     args.results_dir = '/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/results/'
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     args.output_type = 'predictions'

#     data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data/'
#     args.data_dir = os.path.join(data_dir, 'subj'+args.subj)

#     model = brain_encoder_wrapper(args)


