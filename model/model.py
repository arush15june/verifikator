"""
    Verifikator Model
        - Loader for dataset images.
        - Siamese Network Model for training using
          genuine and forged images

    08/11/2018

    TODO:
        - Better image loading
"""
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.utils
import torchvision.transforms as transforms

class Config():

    def __init__(self, images_dir, batch_size, epochs, *args, **kwargs):
        self.images_dir = images_dir
        self.train_batch_size = batch_size
        self.train_number_epochs = epochs

class SignatureDataset(Dataset):
    """ 
    :param imageFolderDataset torchvision.datasets.ImageLoader: Loaded image folder, image information
    :param transform: transforms for the image
    :param should_invert: invert tbe image

    Load the dataset from the 
        <self.imageFolderDataset: torchvision.datasets.ImageLoader>
        DEFAULT DATASET FOLDER: ./dataset/

    - Dataset Format
        File Name: `NFI-XXXYYZZZ`

        `XXX` - ID number of a person who has done the signature. 
		`YY` - Image smaple number.
		`ZZZ` - ID number of person whose signature is in photo.    
    """

    # forged in class 0 loaded by datasets.ImageFolder
    FORGED = 0
    # genuine is class 1 loaded by datasets.ImageFolder
    GENUINE = 1
    
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert

    @property
    def ids(self):
        img_files = self.imageFolderDataset.imgs
        sigs_info = []
        for sig in img_files:
            file_name = os.path.basename(sig[0])
            sig_id, _, _ = self.signatureInfo(file_name)
            sigs_info.append(sig_id)
                
        return set(sigs_info)

    @staticmethod
    def randomSelect(CHOICE_1, CHOICE_2):
        return np.where(np.random.random() > 0.5, CHOICE_1, CHOICE_2)

    def getRandomId(self):
        return random.choice(self.ids)
        
    def getRandomImage(self):
        """
            Load a random image from dataset
        """
        return random.choice(self.imageFolderDataset.imgs)

    def getRandomLabel(self):
        """ 
            Generate random label for selecting a second image
        """
        return self.randomSelect(1, 0)
        
    def getImageSameClass(self):
        """
            Get image tuples of the same class
            - Get a random id, <ID>
            - Load genuine image of the <ID>
            - Load another genuine image of the <ID>

            ex. 
                ID := 001
                image1 := 'NFI-00101001.png'
                image2 := 'NFI-00102001.png'
        """
        sig_id = self.getRandomId()
        print(sig_id)
    
    def getImageDifferentClass(self):
        """
            Get image tuples of different class
            - Get a random id, <ID>
            - Load genuine image of the <ID>
            - Load forged image of the <ID>

            ex. 
                ID := 001
                image1 := 'NFI-00101001.png'
                image2 := 'NFI-00101002.png'
        """
        pass

    @staticmethod
    def signatureInfo(filename):
        """
            signature - signature of this person
            sample - Image Sample No
            done_by - Signature done by
        """
        signature = filename[4:7]
        sample = filename[7:9]
        done_by = filename[9:12]

        return signature, sample, done_by    
    
    @staticmethod
    def getImage(file_name):
        """
            Load file as PIL.Image
        """
        return Image.open(file_name).convert("L")

    @staticmethod
    def generateLabel(label):
        """
            Convert similarity label to PyTorch Tensor
        """
        return torch.from_numpy(np.array([int(label)], dtype=np.float32))

    def __getitem__(self, index):
        """
            Return 2 images and a label corresponding to if the images are the same or not

            - Choose either to select a different image or same image
                - Different Images
                    - Randomly choose a genuine image of a random id
                    - Randomly choose a forged image of the same id
                - Same Images
                    - Randomly choose a genuine image of a random id
                    - Randomly choose another genuine image of the same id
            - Apply Transforms
            - Return 
                - Image 1
                - Image 2
                - Similiarity Label
        """
        
        """ 
            we need to make sure approx 50% of images are in the same class
        """
        should_get_same_class = self.getRandomLabel()

        if should_get_same_class:
            img0_tuple, img1_tuple = self.getImageSameClass()
        else:
            img0_tuple, img1_tuple = self.getImageDifferentClass()

        """
        open images and convert to to grayscale 
        """
        img0 = self.getImage(img0_tuple[0])
        img1 = self.getImage(img1_tuple[0])
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , self.generateLabel(should_get_same_class)

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*130*150, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetworkDataset(Dataset):
    """ 
    Get the dataset for the network, the aim is to get image pairs randomly where 50% 
    image pairs belong to the same class and 50% to a different class
    """
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def getRandomImage(self):
        return random.choice(self.imageFolderDataset.imgs)

    def getRandomLabel(self):
        return np.where(np.random.random() > 0.5, 1, 0)
    
    @staticmethod
    def getImage(file_name):
        return Image.open(file_name)#.convert("L")

    @staticmethod
    def generateLabel(label):
        return torch.from_numpy(np.array([int(label)], dtype=np.float32))

    def __getitem__(self, index):
        img0_tuple = self.getRandomImage()

        """ we need to make sure approx 50% of images are in the same class """
        should_get_same_class = self.getRandomLabel()

        if should_get_same_class:
            img1_tuple = img0_tuple
        else:
            img1_tuple = self.getRandomImage()

        """ open images and convert to to grayscale """
        img0 = self.getImage(img0_tuple[0])
        img1 = self.getImage(img1_tuple[0])
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , self.generateLabel(should_get_same_class)

    def __len__(self):
        return len(self.imageFolderDataset.imgs)