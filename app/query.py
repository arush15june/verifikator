from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from model import SiameseNet

from config import get_config

config, _ = get_config()

class QueryModel():
    """
        Get the confidence score for two PIL images

        Args
            :param model_path str: path to the model to be loaded

        Properties
            :param self.model nn.Module: Model Type to be used for inference. 
            :param self.ckpt: Loaded model information
            
    """
    def __init__(self, model_path, *args, **kwargs):
        """
            - Load model from model_path 
            - Load state to the model
        """
        model = SiameseNet()
        if config.use_gpu:
            model = model.cuda()
        self.model = model
        self.ckpt = torch.load(model_path)
        self.model.load_state_dict(self.ckpt['model_state'])

    @staticmethod
    def resizeImage(image):
        """
            resize image to model parameters.

            Args
                :param image PIL.Image: PIL Image

            Return
                :param image PIL.Image: resized image
        """
        return image.resize((120, 80), Image.ANTIALIAS).convert('L')
    
    def ImageToTensor(self, image):
        """
            resize and transform PIL image to tensor

            Args
                :param image PIL.Image: PIL Image

            Return
                :param tensor torch.Tesnor: PyTorch Tensor
            
        """
        image = self.resizeImage(image)
        transform=transforms.Compose([transforms.ToTensor()])
        return torch.reshape(transform(image), [1, 1, 80, 120])

    def getConfidence(self, tensor0, tensor1):
        """
            get confidence on the two tensors from the model
            and apply sigmoid.

            Args
                :param tensor0 torch.Tensor: transformed image tensor to be used for inference
                :param tensor1 torch.Tensor: transformed image tensor to be used for inference

            Return
                :param output torch.Tensor: confidence tensor    
        """
        x0, x1 = Variable(tensor0), Variable(tensor1)
        if config.use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
        output = self.model(x0, x1)
        output = F.sigmoid(output)

        return output

    def predict(self, img0, img1):
        """ 
            Infer confidence on the two images from the model 
            and return it.

            Args
                :param img0 PIL.Image: first image to be compared
                :param img1 PIL.Image: second image to be compared

            Return
                :param output torch.Tensor: confidence tensor
                                            confidence > 0.5, Similar Pairs
                                            confidence < 0.5, Dissimilar Pairs    
        """
        img0_tensor = self.ImageToTensor(img0)
        img1_tensor = self.ImageToTensor(img1)
        output = self.getConfidence(img0_tensor, img1_tensor)

        return output