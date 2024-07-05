
Our submission to the [Algonauts 2023 challenge](http://algonauts.csail.mit.edu/challenge.html). 

[Challenge Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/9304#results) 
username: hosseinadeli

### Citing our work

Adeli, H., Minni, S., & Kriegeskorte, N. (2023). Predicting brain activity using Transformers. bioRxiv, 2023-08. [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.08.02.551743v1.abstract)]

Please cite our work by using the following BibTeX entry.

``` bibtex
@article{adeli2023predicting,
  title={Predicting brain activity using Transformers},
  author={Adeli, Hossein and Minni, Sun and Kriegeskorte, Nikolaus},
  journal={bioRxiv},
  pages={2023--08},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
``` 
 
[Hossein Adeli](https://hosseinadeli.github.io/)<br />
ha2366@columbia.edu



## update July 2024

Instead of using ROI based decoder queries, I am using a query for each voxel so there is no need for brain areas mapping. That means:

- You need a big GPU for this  
- The model has around 40K decoder tokens (sum of all voxels in the left and the right hemispheres), one for each voxel. So there can be no self-attention operation on the decoder side as the attention matrix would be prohibitively large, just cross attending to the encoder features patches
- The Batch size would have to be very small, I can only go up to 2 on L40 gpus before running into memory issues. 

It improves accuracy across the board. 

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/voxel_level.png" width = 1000> 



## Model Architecture

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/model_architecture.jpg" width = 1000> 

## Training the model

You can train the model using the code below. 

```bash
python main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'streams_inc'
```
The model can accept many parameters for the run. Here the run number is given, the subject number, which encoder layer output should be fed to the decoder ([-1] in this case), and what type of queries should the transformer decoder be using. 

With 'streams_inc', all the vertices are predicted using queries for all the streams. You can use the visualize_results.ipynb to see the results after they are saved. 

Results from a sample run for subj 1:

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/detr_dino_1_streams_inc_16.png" width = 1000> 

In order to train the model using a lower level features from the encoder and to focus on early visual areas:

```bash
python main.py --run 1  --subj 1 --enc_output_layer 8 --readout_res 'visuals'
```


Results from a sample run for subj 1:

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/detr_dino_8_visuals_16.png" width = 1000> 


Final prediction results (from http://algonauts.csail.mit.edu/visualizations_2023/02_hosseinadeli.png):

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/02_hosseinadeli.png" width = 1000> 





## model checkpoints

Download model checkpoint from this [google drive folder](https://drive.google.com/drive/folders/1HMwl3GRm0vZmji4HOz5OeqRc0dp30V4b?usp=sharing):



## Attention maps

```bash
python main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'faces' --save_model 1
```

Save the model check point. attention_maps.ipynb can then be used to visualize the attention weights for the decoder queries (where each query attends to in the image in order to predict an ROI).  

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/attention_maps.png" width = 700> 


<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/attention_maps_face.png" width = 700> 

<!-- ### Repo map

```bash
├── ops                         # Functional operators
    └ ...
├── components                  # Parts zoo, any of which can be used directly
│   ├── attention
│   │    └ ...                  # all the supported attentions
│   ├── feedforward             #
│   │    └ ...                  # all the supported feedforwards
│   ├── positional_embedding    #
│   │    └ ...                  # all the supported positional embeddings
│   ├── activations.py          #
│   └── multi_head_dispatch.py  # (optional) multihead wrap
|
├── benchmarks
│     └ ...                     # A lot of benchmarks that you can use to test some parts
└── triton
      └ ...                     # (optional) all the triton parts, requires triton + CUDA gpu
``` -->
## Credits

The following repositories were used, either in close to original form or as an inspiration:

1) [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) <br/>
2) [facebookresearch/dino](https://github.com/facebookresearch/dino) <br/>
3) [facebookresearch/detr](https://github.com/facebookresearch/detr) <br/>
4) [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) <br/>
