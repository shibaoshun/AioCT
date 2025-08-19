# AioCT: All-in-One Computed Tomography Artifact Reduction
*Baoshun Shi，Chaowei Wang, Ke Jiang*
## Abatract
Computed Tomography (CT) under undersampled measurement conditions and presence of metallic implants will produce various artifacts, thereby compromising both image quality and clinical reliability. Existing deep learning-based artifact reduction methods are typically tailored to a single artifact type, resulting in limited generalization across diverse artifact types and poor adaptability to mixed artifacts. To address these limitations, we propose a residual prompt-based method, AioCT, for all-in-one artifact reduction that can effectively mitigate various types of artifacts. In particular, our method uses the estimation of task-specific residual between the artifact-contaminated image and the underlying image to generate the artifact-specific information, which is then used as semantic priors and task-aware prompts to dynamically guide the artifact reduction network. To enhance the discrimination ability for various types of artifact reduction tasks and structural preservation ability of the entire network, a hybrid optimization strategy consisting of contrastive learning and task-aware optimization is proposed. Specifically, we build the prompt Mamba module and formulate contrastive prompt regularization to enforce clear task boundaries, while use uncertainty estimation enables adaptive supervision through task-aware uncertainty modeling. Furthermore, a transformer-driven dynamic convolution module adaptively modulates convolutional kernels, enhancing multi-scale feature fusion and structural consistency. Extensive evaluations on synthetic and clinical datasets demonstrate superior generalization to various artifacts scenarios even mixed artifacts, as well as improved structural fidelity, outperforming state-of-the-art methods.
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

## Dataset
You can download the testing datasets from the following Baidu Drive link:

📁 `data`

🔗 [https://pan.baidu.com/s/1APDq6wwOAvHLRP8TWI93kw?pwd=2025](https://pan.baidu.com/s/1APDq6wwOAvHLRP8TWI93kw?pwd=2025)

🔑 Password: `2025`

## Pre-trained Models  

📁 `result`

🔗 [https://pan.baidu.com/s/1XGx1EMfzTgGROTdukM9skQ?pwd=2025](https://pan.baidu.com/s/1XGx1EMfzTgGROTdukM9skQ?pwd=2025)

🔑 Password: `2025`

## Results

![image-20250512164910204](https://github.com/shibaoshun/AioCT/blob/main/fig/result.jpg)
