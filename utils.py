from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from config import cfg
import os
import shutil

def get_dataloader(path, shuffle=True):
    """
    create a Dataloader of the images in the folder
    :param path: path containing the images
    :param shuffle: True if shuffle
    :return: the created dataloader
    """
    # the ImageFolder requires at least one subfolder
    for root, directory, files in os.walk(path):
        if root == path:
            # we need to create a subfolder
            os.makedirs(os.path.join(root, '1'), exist_ok=True)
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.join(root, '1', file)
                shutil.move(src, dst)

    transform = transforms.Compose(
        [
            transforms.Resize((cfg.input_size, 2 * cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)
    return loader