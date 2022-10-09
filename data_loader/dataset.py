import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Random90Rot:
    def __init__(self, p=0.5):
        # Rotation probability
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            rot_dir = random.choice([-1, 1])
            x = torch.rot90(x, rot_dir, [1,2])
        return x

class RandomBlur:
    def __init__(self, p=0.1, blur_size=7):
        self.p = p
        self.blur = transforms.GaussianBlur(7)
    
    def __call__(self, x):
        if random.random() <= self.p:
            x = self.blur(x)
        
        return x

class ConditionalResize:
    """Resize a tensor if it less than a given size."""
    def __init__(self, min_size=256):
        self._min_size = min_size
        
    def __call__(self, tensor):
        if tensor.shape[1] < self._min_size or tensor.shape[2] < self._min_size:
            if tensor.shape[1] < tensor.shape[2]:
                new_width = self._min_size
                new_height =  int(tensor.shape[2] / tensor.shape[1] * new_width)
            else:
                new_height = self._min_size
                new_width =  int(tensor.shape[1] / tensor.shape[2] * new_height)
            #t = transforms.Resize([new_width, new_height])
            t = transforms.Resize([new_width, new_height])
            tensor = t(tensor)#F.interpolate(tensor, size=(1, new_width, new_height))

        return tensor 

class RandomResizeOrCrop:
    """Either crop and resize or only resize."""
    def __init__(self, img_size, p=0.50):
        """Args
            p: float giving probability of only resizing.
        """
        self.p = p
        # Resizes to given size
        self.resize = transforms.Resize((img_size[0], img_size[1]))

        # Only resizes if needed
        self.cond_resize = ConditionalResize(min(img_size[0], img_size[1]))
        self.rand_crop_resize = transforms.RandomResizedCrop((img_size[0], img_size[1]))

    def __call__(self, x):
        if random.random() < self.p:
            x = self.resize(x)
        else:
            x = self.cond_resize(x)
            x = self.rand_crop_resize(x)

        return x 
            


class ImageDataset(Dataset):
    def __init__(self, img_paths, class_ids, img_size=256, is_train=False):
        #self.img_paths, self.class_ids = get_paths_and_classes()
        self.img_paths = img_paths
        self.class_ids = class_ids
        if is_train:
            # Use augmentations
            self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            RandomResizeOrCrop((img_size, img_size)),
                                            Random90Rot(0.5),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomVerticalFlip(0.5),
                                            transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 
                                                                                                                0.4, 
                                                                                                                0.4, 
                                                                                                                0.1)], p=0.25),
                                            transforms.RandomGrayscale(p=0.005)]),
                                            RandomBlur(),
                                        ])
        else:
            # Only convert to tensor and resize
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size))
                ])


    def __len__(self):    
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load the image
        img = Image.open(self.img_paths[idx]).convert("RGB")
        
        # Augment the image (if is training data)
        img = self.transforms(img)
        print(self.img_paths[idx], self.class_ids[idx])
        print(img, img.min(), img.max(), img.mean())

        return img, self.class_ids[idx]