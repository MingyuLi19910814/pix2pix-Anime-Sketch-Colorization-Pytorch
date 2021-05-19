from easydict import EasyDict

cfg = EasyDict()
cfg.restart_train = False
cfg.checkpoint_to_load = './ckpt.pt'
cfg.train_folder = '/home/lmy/Desktop/projects/pix2pix/Anime-Sketch-Colorization-Pair/train'
cfg.src_left = False
cfg.test_source_folder = '/home/lmy/Desktop/projects/pix2pix/Anime-Sketch-Colorization-Pair/test'
cfg.test_result_folder = './result'
cfg.batch_size = 32
cfg.input_size = 256
cfg.train_epochs = 100
cfg.d_lr = 0.0002
cfg.g_lr = 0.0002
cfg.l1_lambda = 0
cfg.adam_beta1 = 0.5
cfg.adam_beta2 = 0.999
