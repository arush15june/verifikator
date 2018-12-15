"""
    helpers.py

    - Helper functions for abstracting and handling the model and prediction.

"""
import os
import random

from PIL import Image

import torchvision.datasets as datasets
from torchvision import transforms

from dataset import SignatureDataset, SigInfo
from query import QueryModel

class ModelPredictor():
    """
        Abstract model inference operations

        :params
    """
    def __init__(self, model_path, images_dir):
        self.image_dataset = datasets.ImageFolder(root=images_dir)
        self.queryDataset = SignatureDataset(
                                    dataset=self.image_dataset
                                    )
        self.model = QueryModel(model_path, self.queryDataset)

    @staticmethod
    def getImage(image_file):
        image_file.seek(0)
        image = Image.open(image_file)
        return image

    def _predict(self, img0, img1):
        """
            :param image PIL.Image: PIL Image
        """
        return self.model.predict(img0, img1)

    def getConfidence(self, image_file, cust_id):
        """
            :param image_file BytesIO: File Name/BytesIO Object 
        """
        cust_siginfo_file = "original_{}_1.png".format(cust_id)

        cust_siginfo = SigInfo(cust_siginfo_file)

        sigs_genuine = self.queryDataset.signatures_genuine(cust_siginfo)
        total_sigs = len(sigs_genuine)
        image = self.getImage(image_file)

        output = 0
        for sig in sigs_genuine:
        # sig = random.choice(sigs_genuine)
            sig = Image.open(sig[1])

            confidence = self._predict(sig, image)
            output += confidence

        output = output / total_sigs

        is_genuine = True if confidence > 0.5 else False

        return confidence, is_genuine