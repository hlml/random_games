import torchvision
import torch
import torch.utils.data as utils
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class MNISTCDataset(Dataset):
    
    """MNIST-C dataset."""

    def __init__(self, root, train, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            self.data = torch.tensor(np.load(root + '/train_images.npy'))
            self.targets = torch.tensor(np.load(root + '/train_labels.npy'))
        else:
            self.data = torch.tensor(np.load(root + '/test_images.npy'))
            self.targets = torch.tensor(np.load(root + '/test_labels.npy'))
            
        self.train=train
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][:,:,0], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
