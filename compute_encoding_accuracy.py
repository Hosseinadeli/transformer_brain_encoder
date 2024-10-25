"""Compute the noise-ceiling-normalized encoding accuracy, using NSD's test
split.

Parameters
----------
all_subs : list of int
	List with all NSD subjects.
n_boot_iter : int
	Number of bootstrap iterations for the confidence intervals.
ned_dir : str
	Neural encoding dataset directory.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--all_subs', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--n_boot_iter', default=100000, type=int)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset', type=str)
args = parser.parse_args()

print('>>> Compute encoding accuracy <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Output variables
# =============================================================================
r2 = {}
noise_ceiling = {}
explained_variance = {}


# =============================================================================
# Loop over subjects
# =============================================================================
for sub in tqdm(args.all_subs, leave=False):

# =============================================================================
# Load the fMRI metadata
# =============================================================================
	data_dir = os.path.join(args.ned_dir, 'model_training_datasets',
		'train_dataset-nsd_fsaverage', 'model-algo', 'neural_data',
		'metadata_sub-0'+str(sub)+'.npy')

	metadata = np.load(os.path.join(data_dir, 'metadata_sub-0'+str(sub)+'.npy'),
		allow_pickle=True).item()


# =============================================================================
# Load the biological (ground truth) fMRI responses for the test images
# =============================================================================
	lh_betas = np.load(os.path.join(data_dir, 'lh_betas_sub-0'+str(sub)+'.npy'))
	rh_betas = np.load(os.path.join(data_dir, 'rh_betas_sub-0'+str(sub)+'.npy'))

	# Only keep the fMRI responses for the test images, and average the fMRI
	# responses across the three repeats
	lh_betas_test = np.zeros((len(metadata['test_img_num']),
		lh_betas.shape[1]), dtype=np.float32)
	rh_betas_test = np.zeros((len(metadata['test_img_num']),
		rh_betas.shape[1]), dtype=np.float32)
	for i, img in enumerate(metadata['test_img_num']):
		idx = np.where(metadata['img_presentation_order'] == img)[0]
		lh_betas_test[i] = np.mean(lh_betas[idx], 0)
		rh_betas_test[i] = np.mean(rh_betas[idx], 0)
	del lh_betas, rh_betas


# =============================================================================
# Load the predicted fMRI responses for the test images
# =============================================================================
	# Here you load the fMRI responses for the 515 test images and the subject of
	# interest, predicted by your trained encoding models.
	lh_betas_test_pred = np.load()
	rh_betas_test_pred = np.load()


# =============================================================================
# Convert the ncsnr to noise ceiling
# =============================================================================
	# Left hemisphere
	lh_ncsnr = np.squeeze(metadata['lh_ncsnr'])
	norm_term = (len(lh_betas_test) / 3) / len(lh_betas_test)
	lh_nc = (lh_ncsnr ** 2) / ((lh_ncsnr ** 2) + norm_term)

	# Right hemisphere
	rh_ncsnr = np.squeeze(metadata['rh_ncsnr'])
	norm_term = (len(rh_betas_test) / 3) / len(rh_betas_test)
	rh_nc = (rh_ncsnr ** 2) / ((rh_ncsnr ** 2) + norm_term)


# =============================================================================
# Compute the noise-ceiling-normalized encoding accuracy
# =============================================================================
	# Correlate the biological and predicted data
	lh_correlation = np.zeros(lh_betas_test.shape[1])
	rh_correlation = np.zeros(rh_betas_test.shape[1])
	for v in range(len(lh_correlation)):
		lh_correlation[v] = pearsonr(lh_betas_test[:,v],
			lh_betas_test_pred[:,v])[0]
		rh_correlation[v] = pearsonr(rh_betas_test[:,v],
			rh_betas_test_pred[:,v])[0]
	del lh_betas_test, lh_betas_test_pred, rh_betas_test, rh_betas_test_pred

	# Square the correlation values
	lh_r2 = lh_correlation ** 2
	rh_r2 = rh_correlation ** 2
	r2['s'+str(sub)+'_lh'] = lh_r2
	r2['s'+str(sub)+'_rh'] = rh_r2

	# Add a very small number to noise ceiling values of 0, otherwise
	# the noise-ceiling-normalized encoding accuracy cannot be calculated
	# (division by 0 is not possible)
	lh_nc[lh_nc==0] = 1e-14
	rh_nc[rh_nc==0] = 1e-14
	noise_ceiling['s'+str(sub)+'_lh'] = lh_nc
	noise_ceiling['s'+str(sub)+'_lrh'] = rh_nc

	# Compute the noise-ceiling-normalized encoding accuracy
	expl_var_lh = np.divide(lh_r2, lh_nc)
	expl_var_rh = np.divide(rh_r2, rh_nc)

	# Set the noise-ceiling-normalized encoding accuracy to 1 for those vertices
	# in which the r2 scores are higher than the noise ceiling, to prevent
	# encoding accuracy values higher than 100%
	expl_var_lh[expl_var_lh>1] = 1
	expl_var_rh[expl_var_rh>1] = 1
	explained_variance['s'+str(sub)+'_lh'] = expl_var_lh
	explained_variance['s'+str(sub)+'_rh'] = expl_var_rh


# =============================================================================
# Save the results
# =============================================================================
correlation_results = {
	'r2': r2,
	'noise_ceiling': noise_ceiling,
	'explained_variance': explained_variance,
}

save_dir = os.path.join(args.ned_dir, 'results', 'encoding_accuracy',
	'modality-fmri', 'train_dataset-nsd_fsaverage', 'model-algo')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), correlation_results)
