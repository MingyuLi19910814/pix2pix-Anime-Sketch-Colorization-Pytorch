# Introduction
This repository implements pix2pix with Pytorch and trained with [Anime-Sketch-Colorization-Pair-Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair).
The train image is 256 x 512 (H x W).

# Installation
```bash
git clone https://github.com/MingyuLi19910814/pix2pix-Anime-Sketch-Colorization-Pytorch.git
cd pix2pix-Anime-Sketch-Colorization-Pytorch
conda env create -f environment.yml
conda activate pytorch
```

# Train:
Download the dataset from [Anime-Sketch-Colorization-Pair-Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair) and uncompress it  
```
python train.py --train_dir <train-image-directory> --val_dir <val-image-directory> --train_epochs 100
```
The generated model and images of every epoch will be saved in folder "./train"

# Trained checkpoint

Can be downloaded from https://drive.google.com/file/d/1KqdtD525Y6o1-ng3_9fsx9VLen6r8gL8/view?usp=sharing  
Training detail:  
	Epoch 1 ~ 100: lr = 0.0002  
	Epoch 101 ~ 120: lr = 0.00005  
	GPU: RTX 2080ti. 8min/epoch  

# Colorize sketchs
```
python generate.py --test_dir <test-image-directory> --checkpoint <checkpoint-path>
```
The generated images will be saved in folder "./result"
# Generated samples
![plot](./result/1.jpg)
![plot](./result/2.jpg)
![plot](./result/3.jpg)
![plot](./result/4.jpg)
![plot](./result/5.jpg)
![plot](./result/6.jpg)
![plot](./result/7.jpg)
![plot](./result/8.jpg)
![plot](./result/9.jpg)
