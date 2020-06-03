# ICME2019

by Yuze Guo, SJTU

This project mainly introduce the work of this paper:
[Regularize network skip connections by gating mechanisms for electron microscopy image segmentation]
[https://ieeexplore.ieee.org/document/8784738]

## Introduction 

Recently, one earliest skip connected networks named Lmser was revisited and its convolutional layer based version named CLmser was proposed. This paper studies CLmser for segmentation (shortly CLmser-S) of Electron Microscopy (EM) images and also one further development. First, we experimentally show that CLmser-S outperforms the popular U-Net and save many free parameters. Second, we combine one newest formulation named Flexible Lmser (F-Lmser) and CLmser-S into a version called F-CLmser-S, together with learned masks replacing the similarity based one used in F-Lmser for implementing fast-lane skip connections. Experimental results on the ISBI 2012 EM dataset show that F-CLmser-S improves CLmser and achieves competitive performance with state-of-the-art results.

## Results

We compare our model with some state-of-the-art approaches in ISBI 2012 dataset(http://brainiac2.mit.edu/isbi_challenge/)

|Method | V^{rand} | V^{info} |
|:------|----------|----------|
|SFCNNs | 0.98680  | 0.99144  |
|ADDN   | 0.98317  | 0.99088  |
|**Our**    | 0.98223  | 0.98919  |
|PolyMtl| 0.98058  | 0.98816  |
|M2FCN  | 0.97805  | 0.98919  |
|FusionNet | 0.97804 | 0.98893 |
|CUMedVision | 0.97682 | 0.98865 |
|Unet | 0.97276 | 0.98662 |

## Usage
1. Train and evaluate the model

`CUDA_VISIBLE_DEIVES=0 python train.py --gamma=3 --experiment_idx=0`

2. Testing 

`CUDA_VISIBLE_DEVICES=0 python test.py --gamma=3 --experiment_idx=0 --epoch_idx=1500`

## Citations

If you find our work useful in your research, please consider citing:

`@INPROCEEDINGS{8784738, author={Y. {Guo} and W. {Huang} and Y. {Chen} and S. {Tu}}, booktitle={2019 IEEE International Conference on Multimedia and Expo (ICME)}, title={Regularize Network Skip Connections by Gating Mechanisms for Electron Microscopy Image Segmentation}, year={2019}}`

