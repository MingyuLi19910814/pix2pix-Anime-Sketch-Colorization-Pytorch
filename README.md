# Introduction
This repository implements pix2pix with Pytorch and trained with [Anime-Sketch-Colorization-Pair-Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair).
The train image is 256 x 512 (H x W).

# Set up environment
use Anaconda to create the environment from environment.yml

# Train:
Download the dataset from [Anime-Sketch-Colorization-Pair-Dataset](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair) and uncompress it  
```
python train.py --train_dir <train-image-directory> --val_dir <val-image-directory> --train_epochs 100
```
The generated model and images of every epoch will be saved in folder "./train"
# Colorize sketchs
```
python generate.py --test_dir <test-image-directory>
```
The generated images will be saved in folder "./result"
# Generated samples
![plot](./result/1.jpg)
![plot](./result/2.jpg)
![plot](./result/3.jpg)
