import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
from PIL import ImageFilter
from collections import OrderedDict
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()
# default size
DEFAULT_SIZE = 512, 512


def load_model(model_path, map_location):
    state_dict = torch.load(model_path, map_location=map_location)
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module' for parallel data models
        new_dict[name] = v
    return new_dict


class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(load_model(opt.load_netG_checkpoint_path, cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(load_model(opt.load_netC_checkpoint_path, cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path=None):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        file_dir, file_name = os.path.split(image_path)
        subject_name = os.path.splitext(file_name)[0]
        # Mask
        if mask_path:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.open(image_path).convert('L').point(lambda x: 0 if x < 3 else 255, '1')
            kernel_size = 3
            # dilate mask to fill in small voids
            mask = mask.filter(ImageFilter.MaxFilter(kernel_size))
            # erode mask to restore original size
            mask = mask.filter(ImageFilter.MinFilter(kernel_size))
            mask.save(os.path.join(file_dir, subject_name+'_gen_mask.png'))
        if mask.size != DEFAULT_SIZE:
            mask = self.resize_image(mask, DEFAULT_SIZE,
                                     save_path=os.path.join(file_dir, subject_name+'_gen_mask.png'))
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        if image.size != DEFAULT_SIZE:
            image = self.resize_image(image, DEFAULT_SIZE,
                                      save_path=os.path.join(file_dir, subject_name+'_resized.png'))
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    @staticmethod
    def resize_image(thumbnail, size, save_path=None):
        thumbnail.thumbnail(DEFAULT_SIZE, Image.ANTIALIAS)
        # generating the thumbnail from given size
        thumbnail.thumbnail(size, Image.ANTIALIAS)

        offset_x = int(max((size[0] - thumbnail.size[0]) / 2, 0))
        offset_y = int(max((size[1] - thumbnail.size[1]) / 2, 0))
        offset_tuple = (offset_x, offset_y)  # pack x and y into a tuple

        # create the image object to be the final product
        image = Image.new(mode='RGB', size=size, color=(0, 0, 0))
        # paste the thumbnail into the full sized image
        image.paste(thumbnail, offset_tuple)
        if save_path:
            image.save(save_path)
        return image

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f) and (not 'resized' in f)]
    # test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_images))

    for image_path in tqdm.tqdm(test_images):
        try:
            print(image_path)
            data = evaluator.load_image(image_path)
            evaluator.eval(data, True)
        except Exception as e:
            print("error:", e.args)
