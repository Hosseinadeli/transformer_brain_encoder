from matplotlib import pyplot as plt

import os
os.chdir('/engram/nklab/hossein/recurrent_models/transformer_brain_encoder/')

import numpy as np
import torch.nn.functional as F

import h5py

from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms

from models.resnet import resnet_model
from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

from sklearn.decomposition import IncrementalPCA, PCA
import os
from pathlib import Path

device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
device = torch.device(device)

arch = 'resnet50'

image_size = 425

if 'dino' in arch:
    model = torch.hub.load('facebookresearch/dinov2', arch).to(device)  
    patch_size = 14
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    # #for i in range(1,13):
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

def aff_features(img):

    size_im = (
        img.shape[0],
        img.shape[1],
        int(np.ceil(img.shape[2] / patch_size) * patch_size),
        int(np.ceil(img.shape[3] / patch_size) * patch_size),
    )
    paded = torch.zeros(size_im).to(device)
    paded[:,:, : img.shape[2], : img.shape[3]] = img
    img = paded

    # Size for transformers
    h_featmap = img.shape[-2] // patch_size
    w_featmap = img.shape[-1] // patch_size


    model._modules["blocks"][-10]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    which_features = 'q'

    with torch.no_grad():
        # Forward pass in the model
        outputs = model.get_intermediate_layers(img)

        # Scaling factor
        scales = [patch_size, patch_size]

        # Dimensions
        nb_im = img.shape[0] #Batch size
        nh = 12 #Number of heads
        nb_tokens = h_featmap*w_featmap + 1

        # Extract the qkv features of the last attention layer
        qkv = feat_out["qkv"].reshape(nb_im, nb_tokens, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        # Modality selection
        if which_features == "k":
            feats = k
        elif which_features == "q":
            feats = q
        elif which_features == "v":
            feats = v

        cls_token = feats[0,0:1,:].cpu().numpy() 
        
    #print(feats.flatten(1).dtype)
    return feats.flatten(1).cpu().numpy() 

    #return cls_token[0]

def extract_dino_features(dataloader):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = aff_features(d.to(device))
        # Flatten the features
        features.append(ft)
    return np.vstack(features)


if 'alexnet' in arch:

    model = torch.hub.load('pytorch/vision:v0.10.0', arch)
    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.eval() # set the model to evaluation mode, since you are not training it

    train_nodes, _ = get_graph_node_names(model)
    print(train_nodes)

    #feature_type =  ["features.2"] # "features.2" #"layer2.0.conv1" # #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
    #'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 
    feature_type =  ['features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']
    #feature_type =  ['features.10', 'features.11', 'features.12', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6']

    feature_extractor = create_feature_extractor(model, return_nodes=feature_type).to(device)

def extract_alexnet_features(dataloader):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d.to(device))
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Flatten the features
        features.append(ft.detach().cpu().numpy())
    return np.vstack(features)


if 'resnet' in arch:

    backbone_model = resnet_model('resnet50', train_backbone=False, return_interm_layers=False, dilation=False)
    backbone_model = backbone_model.to(device)

    image_size = 975

    def extract_resnet_features(dataloader):
        features = []
        for _, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            if isinstance(imgs, (list, torch.Tensor)):
                imgs = tuple(imgs.to(device))
                imgs = nested_tensor_from_tensor_list(imgs)

            with torch.no_grad():
                backbone_features = backbone_model(imgs)

            ft = backbone_features['0'].tensors
            ft = torch.hstack([torch.flatten(ft, start_dim=1)])
            # Flatten the features
            features.append(ft.detach().cpu().numpy())
        return np.vstack(features)
    


import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr


data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data'

# device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
# device = torch.device(device)

transform = transforms.Compose([
    transforms.Resize((image_size,image_size)), # resize the images to 224x24 pixels
    transforms.ToTensor(), # convert the images to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
])

for subj in range(1,9): #@param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}

    print(subj)
    class argObj:
        def __init__(self, data_dir, subj):

            self.subj = format(subj, '02')
            self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

    args = argObj(data_dir, subj)

    feature_dir = './saved_image_features/'

    subject_feature_dir =  os.path.join(feature_dir, arch,format(subj, '02'))

    if not os.path.isdir(subject_feature_dir):
        os.makedirs(subject_feature_dir)


    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')
    #test_img_dir = os.path.join(args.data_dir, '../nsdsynthetic_stimuli/')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list = [f for f in train_img_list if f.endswith('.png')]
    train_img_list.sort()

    train_imgs_paths = list(Path(train_img_dir).iterdir())
    train_imgs_paths = [f for f in train_imgs_paths if str(f).endswith('.png')]
    train_imgs_paths = sorted(train_imgs_paths)

    test_img_list = os.listdir(test_img_dir)
    test_img_list = [f for f in test_img_list if f.endswith('.png')]
    test_img_list.sort()

    test_imgs_paths = list(Path(test_img_dir).iterdir())
    test_imgs_paths = [f for f in test_imgs_paths if str(f).endswith('.png')]
    test_imgs_paths = sorted(test_imgs_paths)

    # Create lists with all training and test image file names, sorted
    # train_img_list = os.listdir(train_img_dir)
    # train_img_list.sort()
    # test_img_list = os.listdir(test_img_dir)
    # test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    print('Test images: ' + str(len(test_img_list)))

    idxs_train = np.arange(len(train_img_list))
    idxs_test = np.arange(len(test_img_list))


    class ImageDataset(Dataset):
        def __init__(self, imgs_paths, idxs, transform):
            self.imgs_paths = np.array(imgs_paths)[idxs]
            self.transform = transform

        def __len__(self):
            return len(self.imgs_paths)

        def __getitem__(self, idx):
            # Load the image
            img_path = self.imgs_paths[idx]
            img = Image.open(img_path).convert('RGB')
            # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
            if self.transform:
                img = self.transform(img).to(device)
            return img


    batch_size = 32 #@param
    # Get the paths of all image files
    # train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    # test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform), 
        batch_size=batch_size
    )

    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform), 
        batch_size=batch_size
    )
    
    if 'alexnet' in arch:
        features_train = extract_alexnet_features(train_imgs_dataloader)
        features_test = extract_alexnet_features(test_imgs_dataloader)
    elif 'dino' in arch:
        features_train = extract_dino_features(train_imgs_dataloader)
        features_test = extract_dino_features(test_imgs_dataloader)
    elif 'resnet' in arch:
        features_train = extract_resnet_features(train_imgs_dataloader)
        features_test = extract_resnet_features(test_imgs_dataloader)

    # np.save(subject_feature_dir + '/train.npy', features_train)
    # np.save(subject_feature_dir + '/test.npy', features_test)

    for run in range(1,11):
        print(run)
        save_dir = subject_feature_dir + '/pca_run' + str(run)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # pca = fit_pca(feature_extractor, train_imgs_dataloader)
        # features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)

        num_train = int(np.round(len(features_train) / 100 * 90))
        # Shuffle all training stimulus images
        idxs = np.arange(len(features_train))

        np.random.shuffle(idxs)
        np.save(save_dir+ '/idxs.npy', idxs)
        
        # Assign 90% of the shuffled stimulus images to the training partition,
        # and 10% to the test partition
        idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

        features_train_run = features_train[idxs_train]
        features_val_run = features_train[idxs_val]

        pca = PCA(n_components=768)
        pca.fit(features_train_run)
        features_train_pca = pca.transform(features_train_run)
        features_val_pca = pca.transform(features_val_run)
        features_test_pca = pca.transform(features_test)

        np.save(save_dir + '/train.npy', features_train_pca)
        np.save(save_dir + '/val.npy', features_val_pca)
        np.save(save_dir + '/test.npy', features_test_pca)