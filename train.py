from utils import *
from Network import *

if __name__ == "__main__":
    train_loader = get_dataloader(cfg.train_folder)
    test_loader = get_dataloader(cfg.test_source_folder, shuffle=False)
    model = Model()

    restart_train = False
    last_epoch = 0

    if cfg.restart_train:
        if not os.path.isfile(cfg.checkpoint_to_load):
            restart_train = False
        else:
            last_epoch = model.load(cfg.checkpoint_to_load, load_optim=False)
            restart_train = True
            print('restart train from epoch = {}'.format(last_epoch + 1))

    for epoch in range(last_epoch + 1, cfg.train_epochs + 1):
        model.train_one_epoch(epoch, train_loader)
        model.test_one_epoch(epoch, test_loader)
        model.save(epoch)


