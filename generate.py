from utils import *
from Network import *

if __name__ == "__main__":
    test_loader = get_dataloader(cfg.test_source_folder, shuffle=False)
    model = Model()

    assert os.path.isfile(cfg.checkpoint_to_load)
    last_epoch = model.load(cfg.checkpoint_to_load, load_optim=False)
    restart_train = True
    print('load model from epoch = {}'.format(last_epoch + 1))

    model.generate(test_loader, 'RIGHT')


