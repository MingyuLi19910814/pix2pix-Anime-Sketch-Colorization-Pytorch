This repository implements the 4-layer DCGAN with Pytorch and trained with [Anime-Face-Dataset](https://github.com/bchao1/Anime-Face-Dataset).
The generated images is 64 x 64.

# Set up environment
use Anaconda to create the environment from environment.yml

# Train with your own dataset:
modify "global_cfg.data_path" to your image folder from config.py
```
python train.py
```
The generated model and images of every epoch will be saved in folder "generated_images"
# Generate Anime Faces
```
python inference.py --num_images 128
```
# Generated Samples from Epoch 1 to 150
<img src="demo.gif?raw=true" width="1200px">
