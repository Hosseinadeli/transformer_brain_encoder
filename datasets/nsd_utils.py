


import numpy as np
import torch
import os
from copy import copy
from matplotlib import pyplot as plt
import cortex
import cortex.polyutils

def roi_maps(data_dir):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))



    # lh_challenge_rois_c = np.load('./datasets/lh_challenge_rois_ul.npy')
    # rh_challenge_rois_c = np.load('./datasets/rh_challenge_rois_ul.npy')

    # lh_challenge_rois.append(lh_challenge_rois_c)
    # rh_challenge_rois.append(rh_challenge_rois_c)

    # ul_dict = {0:'uknown'}

    # for i in range(1, np.max(lh_challenge_rois_c)+1):
    #     ul_dict[i] = 'ul_'+str(i)

    #ul_dict = {0:'uknown', 1:'ul_1', 2:'ul_2', 3:'ul_3', 4:'ul_4', 5:'ul_5', 6:'ul_6', 7:'ul_7', 8:'ul_8', 9:'ul_9', 10:'ul_10'}
    #roi_name_maps.append(ul_dict)

    return roi_name_maps, lh_challenge_rois, rh_challenge_rois

def roi_masks(readout_res, roi_name_maps, lh_challenge_rois, rh_challenge_rois):

    if readout_res == 'visuals':
        rois_ind = 0
        num_queries = 16   # 2*len(roi_name_maps[args.rois_ind])

    elif readout_res == 'bodies':
        rois_ind = 1
        num_queries = 16 # 10

    elif readout_res == 'faces':
        rois_ind = 2
        num_queries = 16 #12

    elif readout_res == 'places':
        rois_ind = 3
        num_queries = 16 #8

    elif readout_res == 'words':
        rois_ind = 4
        num_queries = 16 # 12

    elif readout_res == 'streams' or readout_res == 'streams_inc':
        rois_ind = 5
        num_queries = 16

    elif readout_res == 'hemis':
        rois_ind = 5
        num_queries = 2

    elif readout_res == 'voxels':
        rois_ind = [5]

    elif readout_res == 'rois_all':
        rois_ind = [0, 1, 2, 3, 4]

    # elif args.readout_res == 'rois_all_ul':
    #     args.rois_ind = [0, 1, 2, 3, 4, 6]

    lh_challenge_rois_s = []
    rh_challenge_rois_s = []
    lh_roi_names = []
    rh_roi_names = []

    for r in rois_ind:

        #len(roi_name_maps[args.rois_ind])
        #args.roi_nums = len(roi_name_maps[args.rois_ind])

        lh_rois = torch.tensor(lh_challenge_rois[r])
        rh_rois = torch.tensor(rh_challenge_rois[r])

        
        for i in range(1, len(roi_name_maps[r])):
            lh_challenge_rois_s.append(torch.where(lh_rois == i, 1, 0))
            rh_challenge_rois_s.append(torch.where(rh_rois == i, 1, 0))

            lh_roi_names.append(roi_name_maps[r][i])
            rh_roi_names.append(roi_name_maps[r][i])

    lh_challenge_rois_s = torch.vstack(lh_challenge_rois_s)
    rh_challenge_rois_s = torch.vstack(rh_challenge_rois_s)

    lh_challenge_rois_0 = torch.where(lh_challenge_rois_s.sum(0) == 0, 1, 0)
    rh_challenge_rois_0 = torch.where(rh_challenge_rois_s.sum(0) == 0, 1, 0)

    lh_challenge_rois_s = torch.cat((lh_challenge_rois_s, lh_challenge_rois_0[None,:]), dim=0)
    rh_challenge_rois_s = torch.cat((rh_challenge_rois_s, rh_challenge_rois_0[None,:]), dim=0)

    lh_vs = lh_challenge_rois_s.shape[1]
    rh_vs = rh_challenge_rois_s.shape[1]   

    if readout_res == 'voxels':
        num_queries = lh_vs + rh_vs
    elif readout_res == 'rois_all':
        num_queries = lh_challenge_rois_s.shape[0] + rh_challenge_rois_s.shape[0]

    return lh_challenge_rois_s, rh_challenge_rois_s, lh_roi_names, rh_roi_names, num_queries


def plot_on_brain(lh_, rh_, subj=[1]):

    """Plot data on a flattened brain surface using pycortex.

    """
    # =============================================================================
    # Map the data to fsaverage space
    # =============================================================================
    # pycortex requires data in fsaverage space, so here you map the vertices from
    # Challenge space into fsaverage space. The voxels not used in the Challenge
    # are given NaN values, so that pycortex ignores them for the plotting.
    # "ls_scores" and "rh_scores" are lists with 8 elements, one for each subject.
    # These elements consist of vectors of length N, where N is the vertex amount
    # for each subject and hemisphere, and each vector component consists of the
    # prediction accuracy for that vertex.

    #challenge_data_dir = '../algonauts_2023_challenge_data'
    challenge_data_dir = '/engram/nklab/algonauts/algonauts_2023_challenge_data/'
    lh_fsaverage = []
    rh_fsaverage = []
    subjects = subj
    for s, sub in enumerate(subjects):
        lh_mask_dir = os.path.join(challenge_data_dir, 'subj'+format(sub, '02'),
            'roi_masks', 'lh.all-vertices_fsaverage_space.npy')
        rh_mask_dir = os.path.join(challenge_data_dir, 'subj'+format(sub, '02'),
            'roi_masks', 'rh.all-vertices_fsaverage_space.npy')
        lh_fsaverage_all_vertices = np.load(lh_mask_dir)
        rh_fsaverage_all_vertices = np.load(rh_mask_dir)
        lh_fsavg = np.empty((len(lh_fsaverage_all_vertices)))
        lh_fsavg[:] = np.nan
        lh_fsavg[np.where(lh_fsaverage_all_vertices)[0]] = lh_ #lh_scores[s]
        lh_fsaverage.append(copy(lh_fsavg))
        rh_fsavg = np.empty((len(rh_fsaverage_all_vertices)))
        rh_fsavg[:] = np.nan
        rh_fsavg[np.where(rh_fsaverage_all_vertices)[0]] = rh_ #rh_scores[s]
        rh_fsaverage.append(copy(rh_fsavg))
        
        break

    # Average the scores across subjects
    lh_fsaverage = np.nanmean(lh_fsaverage, 0)
    rh_fsaverage = np.nanmean(rh_fsaverage, 0)


    # =============================================================================
    # Plot parameters for colorbar
    # =============================================================================
    plt.rc('xtick', labelsize=19)
    plt.rc('ytick', labelsize=19)

    # =============================================================================
    # Plot the results on brain surfaces
    # =============================================================================
    subject = 'fsaverage'
    data = np.append(lh_fsaverage, rh_fsaverage) * 100
    vertex_data = cortex.Vertex(data, subject, cmap='RdBu_r', vmin=0, vmax=100)
    fig = cortex.quickshow(vertex_data, with_curvature=True)
    return fig
    plt.savefig('my_plot.png', dpi=300) 
    # plt.show()

