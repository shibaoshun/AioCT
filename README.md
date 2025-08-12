# AioCT: All-in-One CT Artifact Reduction
Baoshun Shi，Chaowei Wang, Ke Jiang
## Abatract
![image name](https://github.com/shibaoshun/AioCT/blob/main/fig/AioCT.jpg)
## Installation
The model is built in PyTorch 2.0.1 and  trained with NVIDIA 4090 GPU.
For installing, follow these intructions
```
conda create -n AioCT python=3.10
conda activate AioCT
pip install -r requirements.txt
```
## Install selective_scan_cuda_oflex_rh
[Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba)

## Training and Testing
Training and Testing codes for AioCT are provided in directories.

## Dataset
We train and test our AioCT on Deeplesion. The download links of datasets are provided.
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

