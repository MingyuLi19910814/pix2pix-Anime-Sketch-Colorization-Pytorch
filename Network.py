from torch import nn, optim
import torch
from config import cfg
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import cv2

class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, 4, 2, 1)

        self.conv2 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(dim * 2)

        self.conv3 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(dim * 4)

        self.conv4 = nn.Conv2d(dim * 4, dim * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(dim * 8)

        self.conv5 = nn.Conv2d(dim * 8, dim * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(dim * 8)

        self.conv6 = nn.Conv2d(dim * 8, dim * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(dim * 8)

        self.conv7 = nn.Conv2d(dim * 8, dim * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(dim * 8)

        self.conv8 = nn.Conv2d(dim * 8, dim * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(dim * 8, dim * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(dim * 8)

        self.deconv2 = nn.ConvTranspose2d(dim * 8 * 2, dim * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(dim * 8)

        self.deconv3 = nn.ConvTranspose2d(dim * 8 * 2, dim * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(dim * 8)

        self.deconv4 = nn.ConvTranspose2d(dim * 8 * 2, dim * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(dim * 8)

        self.deconv5 = nn.ConvTranspose2d(dim * 8 * 2, dim * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(dim * 4)

        self.deconv6 = nn.ConvTranspose2d(dim * 4 * 2, dim * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(dim * 2)

        self.deconv7 = nn.ConvTranspose2d(dim * 2 * 2, dim, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(dim)

        self.deconv8 = nn.ConvTranspose2d(dim * 2, 3, 4, 2, 1)


    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2_bn(self.conv2(nn.LeakyReLU(0.2)(e1)))
        e3 = self.conv3_bn(self.conv3(nn.LeakyReLU(0.2)(e2)))
        e4 = self.conv4_bn(self.conv4(nn.LeakyReLU(0.2)(e3)))
        e5 = self.conv5_bn(self.conv5(nn.LeakyReLU(0.2)(e4)))
        e6 = self.conv6_bn(self.conv6(nn.LeakyReLU(0.2)(e5)))
        e7 = self.conv7_bn(self.conv7(nn.LeakyReLU(0.2)(e6)))
        e8 = self.conv8(nn.LeakyReLU(0.2)(e7))

        d1 = nn.Dropout(0.5)(self.deconv1_bn(self.deconv1(nn.ReLU()(e8))))
        d1 = torch.cat([d1, e7], 1)
        d2 = nn.Dropout(0.5)(self.deconv2_bn(self.deconv2(nn.ReLU()(d1))))
        d2 = torch.cat([d2, e6], 1)
        d3 = nn.Dropout(0.5)(self.deconv3_bn(self.deconv3(nn.ReLU()(d2))))
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(nn.ReLU()(d3)))
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(nn.ReLU()(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(nn.ReLU()(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(nn.ReLU()(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(nn.ReLU()(d7))
        return torch.tanh(d8)


class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(6, dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(dim * 4, dim * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(dim * 8)
        self.conv5 = nn.Conv2d(dim * 8, 1, 4, 1, 1)

    def forward(self, x, target):
        x = torch.cat([x, target], 1)
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2_bn(self.conv2(x)))
        x = nn.LeakyReLU(0.2)(self.conv3_bn(self.conv3(x)))
        x = nn.LeakyReLU(0.2)(self.conv4_bn(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        return x

class Model:
    def __init__(self, dim=64):
        self.G = Generator(dim)
        self.D = Discriminator(dim)
        for name, module in self.G._modules.items():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0.0, 0.02)
                module.bias.data.zero_()
        for name, module in self.D._modules.items():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0.0, 0.02)
                module.bias.data.zero_()
        self.D.cuda()
        self.G.cuda()
        self.BCE_loss = nn.BCELoss()
        self.L1_loss = nn.L1Loss()
        self.g_optim = optim.Adam(self.G.parameters(), lr=cfg.g_lr, betas=(cfg.adam_beta1, cfg.adam_beta2))
        self.d_optim = optim.Adam(self.D.parameters(), lr=cfg.d_lr, betas=(cfg.adam_beta1, cfg.adam_beta2))
        self.d_loss = []
        self.g_loss = []

    def load(self, ckpt, load_optim=True):
        """
        Load the checkpoint of the model
        :param ckpt: checkpoint path to load
        :param load_optim: True if load the state dict of optimizers.
        Set as false if you want to change the learning rate.
        :return: the training epoch of the checkpoint
        """
        ckpt = torch.load(ckpt)
        self.D.load_state_dict(ckpt['D'])
        self.G.load_state_dict(ckpt['G'])
        if load_optim:
            self.g_optim.load_state_dict(ckpt['g_optim'])
            self.d_optim.load_state_dict(ckpt['d_optim'])
        return ckpt['epoch']

    def train_one_epoch(self, current_epoch, loader):
        """
        :param current_epoch: current training epoch
        :param loader:  Dataloader of the train samples
        :return: None
        """
        self.D.train()
        self.G.train()
        bar = tqdm(total=loader.__len__())
        d_loss = []
        g_loss = []
        for images, _ in loader:
            images = images.cuda()
            """
            Every image of Anime Sketch Colorization Pair dataset consists of left half part and right half part with 
            same size. The left part is the color image and right part is the corresponding sketch image.
            In this project, we want to generate color image from the sketch image. Therefore, the source image is 
            the right part and target image is the left part. For some other datasets, this maybe different.
            """
            if cfg.src_left:
                source_images = images[:, :, :, :cfg.input_size]
                target_images = images[:, :, :, cfg.input_size:]
            else:
                source_images = images[:, :, :, cfg.input_size:]
                target_images = images[:, :, :, :cfg.input_size]

            self.d_optim.zero_grad()
            self.g_optim.zero_grad()
            real_images_result = self.D(source_images, target_images).squeeze()
            real_images_loss = self.BCE_loss(real_images_result, torch.ones_like(real_images_result))

            fake_images = self.G(source_images)
            fake_images_result = self.D(source_images, fake_images).squeeze()
            fake_images_loss = self.BCE_loss(fake_images_result, torch.zeros_like(fake_images_result))
            D_loss = (real_images_loss + fake_images_loss) * 0.5
            D_loss.backward()
            self.d_optim.step()

            fake_images = self.G(source_images)
            fake_images_result = self.D(source_images, fake_images).squeeze()
            G_loss = self.BCE_loss(fake_images_result, torch.ones_like(fake_images_result)) \
                     + cfg.l1_lambda * self.L1_loss(fake_images, target_images)
            G_loss.backward()
            self.g_optim.step()
            bar.update(1)
            d_loss.append(D_loss.item())
            g_loss.append(G_loss.item())
        bar.close()
        d_loss = np.average(d_loss)
        g_loss = np.average(g_loss)
        print('epoch {} finished. d loss = {}, g loss = {}'.format(current_epoch, d_loss, g_loss))
        self.d_loss.append(d_loss)
        self.g_loss.append(g_loss)

    def test_one_epoch(self, epoch, loader, max_test_num=100):
        """
        Generate the color images of the val sketch samples.
        Save the val images and trained model in  './train/<epoch>'
        :param epoch: training epoch
        :param loader: Dataloader of the val samples
        :param max_test_num: generate at most max_test_num images to save time.
        :return: None
        """
        self.G.eval()
        idx = 0
        folder = os.path.join('train', str(epoch))
        os.makedirs(folder, exist_ok=True)
        for images, _ in loader:
            with torch.no_grad():
                images = images.cuda()
                """
                Every image of Anime Sketch Colorization Pair dataset consists of left half part and right half part with 
                same size. The left part is the color image and right part is the corresponding sketch image.
                In this project, we want to generate color image from the sketch image. Therefore, the source image is 
                the right part and target image is the left part. For some other datasets, this maybe different.
                """
                if cfg.src_left:

                    source_images = images[:, :, :, :cfg.input_size]
                    target_images = images[:, :, :, cfg.input_size:]
                else:
                    source_images = images[:, :, :, cfg.input_size:]
                    target_images = images[:, :, :, :cfg.input_size]
                ims = self.G(source_images).to('cpu').detach().numpy().copy()
                source_images = source_images.to('cpu')
                target_images = target_images.to('cpu')
                ims = np.concatenate([source_images, target_images, ims], axis=3)
                """
                The last function of the generator is torch.tanh, which means the generated image pixels are within [-1, 1].
                To visualize and save the generated images, we need to transfer them into [0, 255].
                """
                ims = (ims + 1) * 0.5
                ims = np.transpose(ims, (0, 2, 3, 1))
                ims = (ims * 255).astype(np.uint8)
                for im in ims:
                    idx += 1
                    image = Image.fromarray(im)
                    if idx < max_test_num:
                        image.save(os.path.join(folder, str(idx) + '.jpg'))

    def generate(self, loader, image_roi='ALL'):
        '''
        :param loader: data loader of the test source images
        :param image_roi: 'LEFT' if left part is used as source image. 'RIGHT' if right part is used. 'ALL' if whole image is used
        :return:
        '''
        assert image_roi in ['LEFT', 'RIGHT', 'ALL']
        self.G.eval()
        idx = 0

        os.makedirs(cfg.test_result_folder, exist_ok=True)
        for images, _ in loader:
            with torch.no_grad():
                if image_roi == 'LEFT':
                    source_images = images[:, :, :, :cfg.input_size]
                    target_images = images[:, :, :, cfg.input_size:]
                elif image_roi == 'RIGHT':
                    source_images = images[:, :, :, cfg.input_size:]
                    target_images = images[:, :, :, :cfg.input_size]
                else:
                    source_images = images
                    target_images = None

                generated_images = self.G(source_images.cuda()).to('cpu').detach().numpy().copy()
                generated_images = (generated_images + 1) * 0.5
                generated_images = np.transpose(generated_images, (0, 2, 3, 1))
                generated_images = (generated_images * 255).astype(np.uint8)

                source_images = source_images.numpy()
                source_images = (source_images + 1) * 0.5
                source_images = np.transpose(source_images, (0, 2, 3, 1))
                source_images = (source_images * 255).astype(np.uint8)

                if target_images is not None:
                    target_images = target_images.numpy()
                    target_images = (target_images + 1) * 0.5
                    target_images = np.transpose(target_images, (0, 2, 3, 1))
                    target_images = (target_images * 255).astype(np.uint8)
                else:
                    target_images = [None] * source_images.shape[0]

                for src_img, tar_img, gen_img in zip(source_images, target_images, generated_images):
                    idx += 1
                    src_img = np.ascontiguousarray(src_img, dtype=np.uint8)
                    src_img = cv2.putText(src_img, 'input', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    if tar_img is not None:
                        tar_img = np.ascontiguousarray(tar_img, dtype=np.uint8)
                        tar_img = cv2.putText(tar_img, 'target', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    gen_img = np.ascontiguousarray(gen_img, dtype=np.uint8)
                    gen_img = cv2.putText(gen_img, 'generated', (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    if tar_img is not None:
                        result_img = np.concatenate([src_img, tar_img, gen_img], axis=1)
                    else:
                        result_img = np.concatenate([src_img, gen_img], axis=1)
                    result_img = Image.fromarray(result_img)
                    result_img.save(os.path.join(cfg.test_result_folder, str(idx) + '.jpg'))

    def save(self, current_epoch, ckpt_path=None):
        """
        Save the trained model state_dict to the check point.
        :param current_epoch: training epoch
        :param ckpt_path: checkpoint path to save
        :return: None
        """
        state = {'G': self.G.state_dict(),
                 'D': self.D.state_dict(),
                 'g_optim': self.g_optim.state_dict(),
                 'd_optim': self.d_optim.state_dict(),
                 'epoch': current_epoch}
        if ckpt_path is None:
            folder = os.path.join('train', str(current_epoch))
            os.makedirs(folder, exist_ok=True)
            ckpt_path = os.path.join(folder, 'ckpt-epoch-{}.pt'.format(current_epoch))
        torch.save(state, ckpt_path)
