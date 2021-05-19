from utils import *
from Network import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process train parameters')
    parser.add_argument('--train_dir', type=str, default=cfg.train_dir)
    parser.add_argument('--val_dir', type=str, default=cfg.val_dir)
    parser.add_argument('--train_epochs', type=int, default=cfg.train_epochs)
    parser.add_argument('--restart_train', type=bool, default=cfg.restart_train)
    parser.add_argument('--checkpoint_to_load', type=str, default=cfg.checkpoint_to_load)
    args = parser.parse_args()

    print('********************* TRAINING PARAMETERS ********************* ')
    for key, value in vars(args).items():
        print('{} = {}'.format(key, value))
    print('*************************************************************** ')

    train_loader = get_dataloader(args.train_dir)
    test_loader = get_dataloader(args.val_dir, shuffle=False)
    model = Model()

    restart_train = args.restart_train
    last_epoch = 0

    if args.restart_train:
        if not os.path.isfile(args.checkpoint_to_load):
            restart_train = False
        else:
            last_epoch = model.load(args.checkpoint_to_load, load_optim=False)
            restart_train = True
            print('restart train from epoch = {}'.format(last_epoch + 1))

    for epoch in range(last_epoch + 1, args.train_epochs + 1):
        model.train_one_epoch(epoch, train_loader)
        model.test_one_epoch(epoch, test_loader)
        model.save(epoch)


