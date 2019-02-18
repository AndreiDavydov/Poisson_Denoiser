import torchvision.transforms as transforms

from torch.utils.data import Dataset
from glob import glob

import torch as th


class Transform(object):
    def __init__(self, train_transforms=True):
        self.train_transforms = train_transforms
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        image_tensor = self.to_tensor(image)
        return image_tensor


class BSDS500(Dataset):
    def __init__(self, path_to_datafolder='./DATASETS/BSDS500/BSDS500_Pois_crops/train/', transforms=None):

        self.transforms = transforms
        self.image_filenames = sorted(glob('{}gt/*.pth'.format(path_to_datafolder)))
        self.noisy_filenames = sorted(glob('{}noisy/*.pth'.format(path_to_datafolder)))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = th.load(self.image_filenames[idx])
        noisy = th.load(self.noisy_filenames[idx])
        
        if self.transforms:
            image, noisy = self.transforms(image), self.transforms(noisy)

        return image, noisy