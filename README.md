# AioCT: All-in-One CT Artifact Reduction
Baoshun Shi，Chaowei Wang, Ke Jiang
## Abatract


![image name](https://github.com/shibaoshun/RepFormer/blob/4ceb46455db4b24fa1f99075d01ca9832735e31e/figs/RepFormer.png)
## Installation
The model is built in PyTorch 1.10.0 and  trained with NVIDIA 2080Ti GPU.
For installing, follow these intructions
```
conda create -n AioCT python=3.10
conda activate AioCT
pip install -r requirements.txt
```
##Install selective_scan_cuda_oflex_rh
[Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba)

## Training and Testing
Training and Testing codes for RSEN, SAM and RepFormer are provided in their respective directories.
+ RSEN: Rain streak estimation network to estimate rain streak information.
+ SAM: Segment anything to estimate image edges.
+ RepFormer: Remove rain streaks using rain streaks and image edges.
## Dataset
We train and test our RepFormer in Rain100L, Rain200L. The download links of datasets are provided.
+ Rain100L: 200 training pairs and 100 testing pairs. Download from [Datasets](https://pan.baidu.com/s/16n5hKHkr2rKlz2kBlI5JSQ?pwd=wxdm).
+ Rain200L: 1800 training pairs and 200 testing pairs. Download from [Datasets](https://pan.baidu.com/s/16n5hKHkr2rKlz2kBlI5JSQ?pwd=wxdm).
## Pre-trained Models  
### For RSEN
Please download checkpoints from [RSEN](https://pan.baidu.com/s/1VyZRqqfCUSZm5zilCIlw9g?pwd=edij).
### For SAM
Please download checkpoints  for the corresponding model type from [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).
### For RepFormer
Please download checkpoints  for the corresponding model type from [RepFormer](https://pan.baidu.com/s/19pubT7KBlKrUbLH19QAERw?pwd=53ws)


## Performance Evaluation 


## Acknowledgements

