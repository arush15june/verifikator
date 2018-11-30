"""
    Modules for the Signature Dataset

    Reference
        https://github.com/kevinzakka/one-shot-siamese

    TODO:
        parameter based SigInfo dataset switching        
"""
import re
import numpy as np
import random
import math
import datetime
from PIL import Image
import torch
import os
import time
from random import Random
import Augmentor

import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_train,
                           augment,
                           way,
                           trials,
                           shuffle=False,
                           seed=0,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid multi-process
    iterators over the Signature dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to load the augmented version of the train dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_dataset = dset.ImageFolder(root=train_dir)
    train_dataset = SignatureDataset(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_dataset = dset.ImageFolder(root=valid_dir)
    valid_dataset = SignatureTest(
        dataset=valid_dataset, trials=trials, way=way, seed=seed,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    way,
                    trials,
                    seed=0,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process iterator
    over the Omniglot test dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = dset.ImageFolder(root=test_dir)
    test_dataset = SignatureTest(
        test_dataset, trials=trials, way=way, seed=seed,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=way, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return test_loader

class SigInfo(object):
    """
        Signature metadata from filename
    """
    
    DEFAULT_DATASET_TYPE = 'cedar'
    
    DATASET_TYPES = {
        # original_1_1.png
        'cedar': re.compile(r'(?P<genuine>original|forgeries)_(?P<signature>\d+)_(?P<sample>\d+).*'),
        # 00101001.png
        'sample': re.compile(r'(?P<signature>\d{3})(?P<sample>\d{2})(?P<done_by>\d{3}).*'),
        # H-S-2-F-01.tiff
        'bhsig': re.compile(r'[H|B]-[S]-(?P<signature>\d+)-(?P<genuine>F|G)-(?P<sample>\d+).*')
    }
    
    def __init__(self, signature_filename, *args, **kwargs):
        """
            Get signature file metadata from filename string
            
            Properties
                - signature - signature of this person
                - sample - Image Sample No
                - done_by - Signature done by
        """

        # Assume Forged
        self.genuine = False
        self.forged = True

        # Signature and Sample common accross all datasets
        self.signature = None
        self.sample = None

        # Set dataset type and set metadata
        # self.dataset_type = 'cedar'
        # self._set_cedar(signature_filename)
        self._set_dataset_type(signature_filename)
        
    def __repr__(self):
        return "<SigInfo {} {} {} {}>".format(
            str(self.signature).zfill(3),
            str(self.sample).zfill(3),
            'G' if self.genuine else 'F',
            self.dataset_type
        )

    def _set_cedar(self, signature_filename):
        filename_without_ext = signature_filename[:-4]
        filename_split = filename_without_ext.split('_')
        genuine = True if filename_split[0] == 'original' else False

        self.signature = filename_split[1]
        self.sample = filename_split[2]
        
        if genuine:
            self.genuine = True
            self.forged = False

    def _set_dataset_type(self, signature_filename):
        for dataset_type, pattern in self.DATASET_TYPES.items():
            pattern_match = pattern.match(signature_filename)
            if pattern_match:
                self.dataset_type = dataset_type
                self._set_sig_metadata(pattern_match)
                break
                
    def _set_sig_metadata(self, matched_pattern):
        eval('self._{}_set_metadata(matched_pattern)'.format(self.dataset_type))
                
    def _sample_set_metadata(self, matched_pattern):
        self.signature = int(matched_pattern.group('signature'))
        self.sample = int(matched_pattern.group('sample'))
        self.done_by = int(matched_pattern.group('done_by'))

        if(self.signature == self.done_by):
            self.genuine = True
            self.forged = False

    def _cedar_set_metadata(self, matched_pattern):
        self.signature = int(matched_pattern.group('signature'))
        self.sample = int(matched_pattern.group('sample'))
        genuine = True if matched_pattern.group('genuine') == 'original' else False

        if genuine:
            self.genuine = True
            self.forged = False
    
    def _bhsig_set_metadata(self, matched_pattern):
        self.signature = int(matched_pattern.group('signature'))
        self.sample = int(matched_pattern.group('sample'))
        genuine = True if matched_pattern.group('genuine') == 'G' else False
        
        if genuine:
            self.genuine = True
            self.forged = False

class SignatureTools(object):
    """
        Tools to work on SigInfo instances.
    """
    @property
    def signatures(self):
        """
            Generate list of all signature metadata and filepaths
        """
        img_files = self.dataset.imgs
        signatures = []
        for sig in img_files:
            file_path = sig[0]
            file_name = os.path.basename(sig[0])
            sig_info = SigInfo(file_name)
            signatures.append((sig_info, file_path))
                
        return signatures

    def signatures_id(self, query_sig):
        """
            Get all signatures (genuine + forged)
            from `self.signatures` 
            belonging to a specific id
        """
        return list(filter(lambda sig_info: sig_info[0].signature == query_sig.signature, self.signatures))

    def signatures_exclude(self, query_sig):
        """
            Get all signatures (genuine + forged)
            from `self.signatures`
            not belonging to a specific id
        """
        return list(filter(lambda sig_info: sig_info[0].signature != query_sig.signature, self.signatures))

    @property
    def genuine(self):
        """
            Return list of all genuine signatures from all signatures
        """
        return list(filter(lambda sig_info: sig_info[0].genuine, self.signatures))

    @property
    def forged(self):
        """
            Return list of all forged signatures from all signatures
        """
        return list(filter(lambda sig_info: sig_info[0].forged, self.signatures))

    def signatures_genuine(self, query_sig):
        """
            Return list of all genuine signatures of <query_sig>'s ID
        """
        signature_list = self.signatures_id(query_sig)
        return list(filter(lambda sig_info: sig_info[0].genuine, signature_list))

    def signatures_forged(self, query_sig):
        """
            Return list of all forged signatures of <query_sig>'s ID
        """
        signature_list = self.signatures_id(query_sig)
        return list(filter(lambda sig_info: sig_info[0].forged, signature_list))

class SignatureDataset(Dataset, SignatureTools):
    """
    Handles the signature dataset images

    Args
    ----

    :param torchvision.datasets.ImageLoader dataset: Loaded image folder, image information
    :param int num_train: number of training samples
    :param bool augment: augment images to be returned or not

    Load the dataset from the 
        <self.imageFolderDataset: torchvision.datasets.ImageLoader>
        DEFAULT DATASET FOLDER: `./dataset/`

    - Dataset Format
        Dataset contains forged/ and geunine/ images
    
        File Name: `NFI-XXXYYZZZ`

        `XXX` - ID number of a person who has done the signature. 
		`YY` - Image smaple number.
		`ZZZ` - ID number of person whose signature is in photo.

    Return 2 Photos and a corresponding similarity label.    
    """
    def __init__(self, dataset, augment=False, *args, **kwargs):
        super(SignatureDataset, self).__init__()
        self.dataset = dataset
        self.augment = augment
        self.dataset_type = kwargs.get('dataset_type')

    def __len__(self):
        """
            genuine signatures represent the list of
            valid signatures to be trained on.
        """
        return len(self.genuine)

    def __getitem__(self, index):
        """
            - Select an image from the list of genuine signatures
            - Get all genuine and forged samples for the corresponding id
            
            - Generate label randomly
                - label = random.choice([1.0, 0.0])

            - if label is 1.0
                - Select an image randomly from the list of genuine images
                    - Fallback, if there 
            - if label is 0.0
                - Select an image randomly from list of forged images

            acquired: 
                image1: File Path String
                image2: File Path String
                label: Boolean

            - Load images using PIL

            - Augment the first image if required
            
            - Cast images to Tensors

            - Return
                :param image1 Tensor: First image as `Tensor`
                :param image2 Tensor: Second image as `Tensor`
                :param label Tensor: Class label for similar/dissimlar classes
        """
        image1 = self.genuine[index] # (signature, sample_no, done_by, file_path)
        image1_filename = os.path.basename(image1[1])
        
        image1_sig_meta = SigInfo(image1_filename, dataset_type=self.dataset_type)

        label = random.choice([1.0, 0.0])
        # get image from same class
        if label:
            sigs_genuine = self.signatures_genuine(image1_sig_meta)
            image2 = random.choice(sigs_genuine)
        # get image from different class
        else:
            sigs_forged = self.signatures_forged(image1_sig_meta)
            if len(sigs_forged) > 0:
                image2 = random.choice(sigs_forged)
            else:
                sigs_exclude = self.signatures_exclude(image1_sig_meta)
                image2 = random.choice(sigs_exclude)

        image1 = Image.open(image1[1]).resize((96, 64))
        image2 = Image.open(image2[1]).resize((96, 64))
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # apply transformation on the fly
        if self.augment:
            p = Augmentor.Pipeline()
            p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
            p.random_distortion(
                probability=0.5, grid_width=6, grid_height=6, magnitude=10,
            )
            trans = transforms.Compose([
                p.torch_transform(),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.ToTensor()

        image1 = trans(image1)
        image2 = transforms.ToTensor()(image2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (image1, image2, y)