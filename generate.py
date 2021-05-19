from utils import *
from Network import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process generate parameters')
    parser.add_argument('--test_dir', type=str, default=cfg.test_dir)
    parser.add_argument('--checkpoint', type=str, default=cfg.checkpoint_to_load)
    args = parser.parse_args()

    print('********************* GENERATING PARAMETERS ********************* ')
    for key, value in vars(args).items():
        print('{} = {}'.format(key, value))
    print('*************************************************************** ')
    test_loader = get_dataloader(args.test_dir, shuffle=False)
    model = Model()

    assert os.path.isfile(args.checkpoint)
    last_epoch = model.load(args.checkpoint, load_optim=False)
    restart_train = True
    print('load model from epoch = {}'.format(last_epoch))

    model.generate(test_loader, 'RIGHT')


