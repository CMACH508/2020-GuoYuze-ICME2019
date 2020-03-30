# ICME2019
Code for paper in ICME2019, "Regularize network skip connections by gating mechanisms for electron microscopy image segmentation"

## Usage
1. Train and evaluate the model

`CUDA_VISIBLE_DEIVES=0 python train.py --gamma=3 --experiment_idx=0`

2. Testing 

`CUDA_VISIBLE_DEVICES=0 python test.py --gamma=3 --experiment_idx=0 --epoch_idx=1500`

