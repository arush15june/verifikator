from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from model import SiameseNet

class QueryModel():
    """
        Get top 10 similar products from the database by passing a PIL.Image
    """
    def __init__(self, model_path, *args, **kwargs):
        self.model = SiameseNet().cuda()
        self.model_path = model_path
        self.ckpt = torch.load(model_path)
        self.model.load_state_dict(self.ckpt['model_state'])

    @staticmethod
    def resizeImage(image):
        return image.resize((96, 64), Image.ANTIALIAS)
    
    def ImageToTensor(self, image):
        image = self.resizeImage(image)
        transform=transforms.Compose([transforms.ToTensor()])
        return torch.reshape(transform(image), [1, 1, 64, 96])

    def getDissimilarity(self, tensor0, tensor1):
        output = self.model(Variable(tensor0).cuda(), Variable(tensor1).cuda())
        
        return output

    def predict(self, image1, image2):
        """ image: PIL Image """
        image1_tensor = self.ImageToTensor(image1)
        image2_tensor = self.ImageToTensor(image2)
        output = self.getDissimilarity(image1_tensor, image2_tensor)
        return output