from easydict import EasyDict

cfg = EasyDict()
# restart the training from some checkpoint
cfg.restart_train = False
# the checkpoint for restaring the training
cfg.checkpoint_to_load = './ckpt.pt'
# directory of train samples
cfg.train_dir = '/home/lmy/Desktop/projects/pix2pix/Anime-Sketch-Colorization-Pair/train'
# directory of val samples
cfg.val_dir = '/home/lmy/Desktop/projects/pix2pix/Anime-Sketch-Colorization-Pair/test'
# set as True if the left half part of the train image is the source (sketch image) and right half part is the target (colorized image)
cfg.src_left = False
# directory of test samples
cfg.test_dir = '/home/lmy/Desktop/projects/pix2pix/Anime-Sketch-Colorization-Pair/test'
# directory to save the generated images of the test samples
cfg.test_result_folder = './result'
# training batch size
cfg.batch_size = 32
# training image size
cfg.input_size = 256
# training epochs
cfg.train_epochs = 100
# discriminator learning rate
cfg.d_lr = 0.00005
# generator learning rate
cfg.g_lr = 0.00005
# generator l1 loss factor
cfg.l1_lambda = 0
# adam beta1
cfg.adam_beta1 = 0.5
# adam beta2
cfg.adam_beta2 = 0.999
