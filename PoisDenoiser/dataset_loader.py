import torchvision.transforms as transforms

from torch.utils.data import Dataset
from glob import glob

import torch as th

from PoisDenoiser.VST import VST_forward


class BSDS500(Dataset):
    def __init__(self, path_to_datafolder='./DATASETS/BSDS500/BSDS500_Pois_crops/train/', \
        num_images=None, do_VST=False, do_VST_4_visual=False, size=None, get_name=False):

        self.size=size
        self.do_VST = do_VST
        self.do_VST_4_visual = do_VST_4_visual
        self.image_filenames = sorted(glob('{}gt/*.pth'.format(path_to_datafolder)))
        self.noisy_filenames = sorted(glob('{}noisy/*.pth'.format(path_to_datafolder)))
        self.path_to_datafolder = path_to_datafolder
        self.get_name = get_name

        if num_images is not None:
            self.image_filenames = self.image_filenames[:num_images]
            self.noisy_filenames = self.noisy_filenames[:num_images]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = th.load(self.image_filenames[idx])
        noisy = th.load(self.noisy_filenames[idx])
        file_name = self.image_filenames[idx][len(self.path_to_datafolder)+3:-4] # 3 == len('gt/'). Used only during Val.

        if self.size is not None:
            image = image[:,:self.size[0],:self.size[1]]
            noisy = noisy[:,:self.size[0],:self.size[1]]
        
        if not self.do_VST and not self.do_VST_4_visual:
            out = [image, noisy]
        if self.do_VST:
            out = [VST_forward(image), VST_forward(noisy)]
        if self.do_VST_4_visual:
            out = [image, VST_forward(noisy), noisy]

        if self.get_name:
            return out+[file_name]
        else:
            return out


class FMD(Dataset):
    def __init__(self, path_to_datafolder='./DATASETS/FMD/fmd/', exp='twophoton', trainorval='train',\
        num_images=None, get_name=True, size=None):

        self.size=size
        self.path_to_datafolder = path_to_datafolder + exp + '/'+trainorval + '/'
        self.image_filenames = sorted(glob('{}gt/*.pth'.format(self.path_to_datafolder)))
        self.noisy_filenames = sorted(glob('{}noisy/*.pth'.format(self.path_to_datafolder)))
        
        self.get_name = get_name

        if num_images is not None:
            self.image_filenames = self.image_filenames[:num_images]
            self.noisy_filenames = self.noisy_filenames[:num_images]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = th.load(self.image_filenames[idx])
        noisy = th.load(self.noisy_filenames[idx])
        file_name = self.image_filenames[idx][len(self.path_to_datafolder):-4]

        if self.size is not None:
            image = image[:,:self.size[0],:self.size[1]]
            noisy = noisy[:,:self.size[0],:self.size[1]]
        
        out = [image, noisy]

        if self.get_name:
            return out+[file_name]
        else:
            return out