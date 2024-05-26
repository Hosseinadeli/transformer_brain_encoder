#!/usr/bin/env python

import collections
import os

import numpy as np
from PIL import Image, ImageOps 
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms
from tqdm import tqdm
import pandas as pd
import csv

def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict

class VGGFaces2(data.Dataset):

    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt

    def __init__(self, args, split='train', num_cats=0, starting_cat_label=0, just_one_cat = False, 
                 transform=None, upper=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param id_label_dict: X[class_id] -> label
        :param split: train or valid
        :param transform: 
        :param horizontal_flip:
        :param upper: max number of image used for debug
        """

        # 0. id label map
        meta_file = '/engram/nklab/VGGFace2/VGG-Face2/meta/identity_meta.csv' 
        #'/scratch/nklab/projects/face_proj/datasets/VGGFace2/meta/identity_meta_cleaned.csv'
        self.id_label_dict = get_id_label_map(meta_file)

        # 1. data loader
        root = '/scratch/nklab/projects/face_proj/datasets/VGGFace2/'

        # '/scratch/nklab/projects/face_proj/models/vgg16_bn/VGGFace2/'
        train_img_list_file = root+ 'meta/train_list.txt' # args.train_img_list_file
        test_img_list_file = root  + 'meta/test_list.txt' # args.test_img_list_file

        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        # assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        # self.image_list_file = image_list_file
        self.split = split

        # TODO should the test be on gray? 
        self.img_channels = args.img_channels
        self.transform = True
        self.transform_gray =  torchvision.transforms.Compose([
    #         transforms.RandomRotation(degrees=(0, 15)),
    #         transforms.RandomCrop(375),
    #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
            torchvision.transforms.ToTensor(), # convert the images to a PyTorch tensor
    #        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])
        
        self.transform_rgb =  torchvision.transforms.Compose([
    #         transforms.RandomRotation(degrees=(0, 15)),
    #         transforms.RandomCrop(375),
    #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
            torchvision.transforms.ToTensor(), # convert the images to a PyTorch tensor
            torchvision.transforms.Normalize([0.6068, 0.4517, 0.3800], [0.2492, 0.2173, 0.2082]) # normalize the images color channels
        ])

        self.horizontal_flip = args.horizontal_flip

        self.image_size = args.image_size
        
        split_folder = self.split
        if self.split == 'val':
            split_folder = 'train'
            
        split_dir = root + split_folder + '/'
            
        dir_list = np.sort(os.listdir(split_dir))
        
        self.img_info = []

        if not num_cats:
            num_cats = len(dir_list)
        
        #print(f'number of categories" {len(dir_list)}')
            
        if split_folder == 'train':
            #im_list = np.random.permutation(im_list)
            
            train_bb_path = '/scratch/nklab/projects/face_proj/datasets/VGGFace2/meta/loose_bb_train_wlabels.csv'
            bb_df = pd.read_csv(train_bb_path)
            
            for i in tqdm(range(num_cats)): #range(len(dir_list))): #range(10): #[1]: #8631

                class_id = dir_list[i]

                im_list = np.sort(os.listdir(split_dir+class_id))
                
                if self.split == 'train':
                    range_beg = 0
                    range_end = int(0.9*len(im_list))
                
                elif self.split == 'val':
                    range_beg = int(0.9*len(im_list))
                    range_end = len(im_list)
                    

                for im_ch in range(range_beg, range_end): # np.random.choice(len(im_list), 1)[0]

                    src_im = im_list[im_ch]

                    if src_im[:2] == '._' : 
                        src_im = src_im[2:]

                    img_file = os.path.join(split_folder, class_id, src_im)

                    label = self.id_label_dict[class_id] + starting_cat_label

                    if just_one_cat:
                        label = starting_cat_label

                    n_id = class_id +'/'+src_im.split('.')[0]
#                     info = train_bb_df[train_bb_df['NAME_ID'] == n_id].iloc[0]
#                     X,Y,bb_w,bb_h= info.X, info.Y, info.W, info.H
#                     bb = (X, Y, bb_w, bb_h)   

                    self.img_info.append({
                        'NAME_ID': n_id,
                        'cid': class_id,
                        'img': img_file,
                        'lbl': label,
#                         'bb': bb,
                    })
                    
            self.img_info = pd.DataFrame(self.img_info)
            self.img_info = self.img_info.merge(bb_df, on='NAME_ID', how='left')

                    
        elif split_folder == 'test':

            test_bb_path = '/scratch/nklab/projects/face_proj/datasets/VGGFace2/meta/loose_bb_test_wlabels.csv'
            bb_df = pd.read_csv(test_bb_path)

            for i in tqdm(range(len(dir_list))): #range(len(dir_list))): #range(10): #[1]: #8631

                class_id = dir_list[i]

                im_list = os.listdir(split_dir+class_id)
        
                for im_ch in range(len(im_list)): # np.random.choice(len(im_list), 1)[0]

                    src_im = im_list[im_ch]
                    img_file = os.path.join(split_folder, class_id, src_im)

                    label = self.id_label_dict[class_id]

                    n_id = class_id +'/'+src_im.split('.')[0]
    #                 info = bb_df[bb_df['NAME_ID'] == n_id].iloc[0]
    #                 X,Y,bb_w,bb_h= info.X, info.Y, info.W, info.H
    #                 bb = (X, Y, bb_w, bb_h)

                    self.img_info.append({
                        'NAME_ID': n_id,
                        'cid': class_id,
                        'img': img_file,
                        'lbl': label,
    #                     'bb': bb,
                    })
        
            self.img_info = pd.DataFrame(self.img_info)
            self.img_info = self.img_info.merge(bb_df, on='NAME_ID', how='left')
                

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info.iloc[index]
        img_file = info['img']
        img = Image.open(os.path.join(self.root, img_file))
        img_w, img_h = img.size
        img = torchvision.transforms.Resize(self.image_size)(img)  #256
        img_re_w, img_re_h = img.size
        #print(f'img before w, h : {img_w}, {img_h}   img after w, h : {img.size}')
        
        if self.split == 'train':
            img = torchvision.transforms.CenterCrop(self.image_size)(img)  #RandomCrop  #CenterCrop
            img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
            if self.horizontal_flip:
                img = torchvision.transforms.functional.hflip(img)
        else:
            img = torchvision.transforms.CenterCrop(self.image_size)(img)
            

        if (self.img_channels == 1) and (img.mode == "RGB"):
            img = ImageOps.grayscale(img)

        img_c_w, img_c_h = img.size
    
        img_w_scale = img_re_w/img_w
        img_h_scale = img_re_h/img_h
        
        img_w_shift = int( (img_re_w - img_c_w) / 2 )
        img_h_shift = int( (img_re_h - img_c_h) / 2 )

        img = np.array(img, dtype=np.uint8)

        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=2)

        assert len(img.shape) == 3  # assumes color images and no alpha channel
        
        patch_size = 14

        label = info['lbl']
        class_id = info['cid']
        if self.transform:
            if self.img_channels == 3:
                img =  self.transform_rgb(img) #self.transform_img(img) #
            elif self.img_channels == 1:
                img =  self.transform_gray(img)

        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / patch_size) * patch_size),
            int(np.ceil(img.shape[2] / patch_size) * patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded
        
        im_h = img.shape[1]
        im_w = img.shape[2]

        X, Y, bb_w, bb_h = info.X, info.Y, info.W, info.H
        
        X, Y, bb_w, bb_h = img_w_scale*X - img_w_shift, img_h_scale*Y - img_h_shift, img_w_scale*bb_w, img_h_scale*bb_h

        bb = torch.tensor([X/im_w, Y/im_h, (X+bb_w)/im_w, (Y+bb_h)/im_h]) 
        #print('bb before', info)
        
        bb = torch.clip(bb, min=0.0, max=1.0)
        #print(bb)
        target = {}
        target["boxes"] = [bb]
        target["labels"] = [label]

        return img, label  #, img_file, class_id
        

    def transform_img(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform_img(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl

