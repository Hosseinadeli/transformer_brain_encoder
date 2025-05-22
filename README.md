

# Transformer brain encoder 

Transformer brain encoders explain human high-level visual responses (https://hosseinadeli.github.io/)


[Hossein Adeli](https://hosseinadeli.github.io/)<br />
ha2366@columbia.edu

## Model Architecture

<img src="https://raw.githubusercontent.com/Hosseinadeli/transformer_brain_encoder/main/figures/arch.png" width = 1000> 

## Training the model

You can train the model using the code below. 

```bash
python main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'rois_all'
```
Here the run number is given, the subject number, which encoder layer output should be fed to the decoder ([-1] in this case), and what type of queries should the transformer decoder be using. 

With 'rois_all', all the vertices are predicted using queries for all the brain areas (so no need for training separate models). You can use the visualize_results.ipynb to see the results after they are saved. 

Results from a sample run for subj 1:

<img src="https://raw.githubusercontent.com/Hosseinadeli/transformer_brain_encoder/main/figures/rois.png" width = 1000> 



### References 

We intially explored this solution in our our submission to the [Algonauts 2023 challenge](http://algonauts.csail.mit.edu/challenge.html). 

[Challenge Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/9304#results) 
username: hosseinadeli

### Citing our work

Adeli, H., Minni, S., & Kriegeskorte, N. (2023). Predicting brain activity using Transformers. bioRxiv, 2023-08. [[bioRxiv](https://www.biorxiv.org/content/10.1101/2023.08.02.551743v1.abstract)]

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
 


## Attention maps



<img src="https://raw.githubusercontent.com/Hosseinadeli/transformer_brain_encoder/main/figures/att.png" width = 700> 

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
