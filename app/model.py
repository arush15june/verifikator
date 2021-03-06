"""
    Verifikator Model
        - Loader for dataset images.
        - Siamese Network Model for training using
          genuine and forged images

    08/11/2018

    Reference
        https://github.com/kevinzakka/one-shot-siamese

    TODO:
        - Better image loading
        - Convert Signature Info to class
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

class SiameseNet(nn.Module):
    """
    A Convolutional Siamese Network for One-Shot Learning [1].

    Siamese networts learn image representations via a supervised metric-based
    approach. Once tuned, their learned features can be leveraged for one-shot
    learning without any retraining.

    References
    ----------
    - Koch et al., https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        # self.conv1_bn = nn.BatchNorm2d(64)
        # self.conv2_bn = nn.BatchNorm2d(128)
        # self.conv3_bn = nn.BatchNorm2d(128)
        # self.conv4_bn = nn.BatchNorm2d(256)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in')
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal(m.weight, 0, 1e-2)
        #         nn.init.normal(m.bias, 0.5, 1e-2)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal(m.weight, 0, 2e-1)
        #         nn.init.normal(m.weight, 0, 1e-2)

    def sub_forward(self, x):
        """
        Forward pass the input image through 1 subnetwork.

        Args
        ----
        - x: a Variable of size (B, C, H, W). Contains either the first or
          second image pair across the input batch.

        Returns
        -------
        - out: a Variable of size (B, 4096). The hidden vector representation
          of the input vector x.
        """
        # out = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        # out = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        # out = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(out))), 2)
        # out = self.conv4_bn(F.relu(self.conv4(out)))
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))
        return out

    def forward(self, x1, x2):
        """
        Forward pass the input image pairs through both subtwins. An image
        pair is composed of a left tensor x1 and a right tensor x2.

        Concretely, we compute the component-wise L1 distance of the hidden
        representations generated by each subnetwork, and feed the difference
        to a final fc-layer followed by a sigmoid activation function to
        generate a similarity score in the range [0, 1] for both embeddings.

        Args
        ----
        - x1: a Variable of size (B, C, H, W). The left image pairs along the
          batch dimension.
        - x2: a Variable of size (B, C, H, W). The right image pairs along the
          batch dimension.

        Returns
        -------
        - probas: a Variable of size (B, 1). A probability scalar indicating
          whether the left and right input pairs, along the batch dimension,
          correspond to the same class. We expect the network to spit out
          values near 1 when they belong to the same class, and 0 otherwise.
        """
        # encode image pairs
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        # compute l1 distance
        diff = torch.abs(h1 - h2)

        # score the similarity between the 2 encodings
        scores = self.fc2(diff)

        # return scores (without sigmoid) and use bce_with_logits
        # for increased numerical stability
        return scores