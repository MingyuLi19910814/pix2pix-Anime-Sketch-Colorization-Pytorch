import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from config import cfg
import os
import shutil
import cv2
from PIL import Image

def get_dataloader(path, shuffle=True):
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


def ExtractVideoBasicInformation(video_path):
    video = cv2.VideoCapture(video_path)
    img_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
    img_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    img_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    img_fps = video.get(cv2.CAP_PROP_FPS)
    return int(img_cnt), img_fps, int(img_width), int(img_height)

def ExtractImageFramesFromVideo(video_path):
    video = cv2.VideoCapture(video_path)
    while True:
        success, image = video.read()
        if not success:
            break
        yield image
