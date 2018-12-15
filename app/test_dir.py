import os
import pandas as pd
import torchvision.datasets as datasets
from PIL import Image

from config import get_dir_test_config

from query import QueryModel
from dataset import SigInfo

class Tester(object):

    DEFAULT_MODEL_PATH = '../model/ckpt/exp_13/model_ckpt.tar'
    CSV_FILENAME = 'output.csv'

    def __init__(self, id_dir, data_dir, *args, **kwargs):
        self.id_dataset = datasets.ImageFolder(root=id_dir)
        self.test_dataset = datasets.ImageFolder(root=data_dir)
        model_path = kwargs.get('model_path', self.DEFAULT_MODEL_PATH)
        self.query_model = QueryModel(model_path) 
        self.test_output = None
        self._test_data()

    def id_images(self, id):
        """
            Return all images for an ID from the id dataset
        """
        id_sigs = [{'metadata': SigInfo(os.path.basename(id_file[0])), 'img': id_file[0]} for id_file in self.id_dataset.imgs]
        return list(filter(lambda sig: sig['metadata'].signature == id, id_sigs))

    def _test_id(self, query_image, id):
        images = self.id_images(id)
        total_images = len(images)
        total_confidence = 0

        for sig in images:
            id_image = Image.open(sig['img'])
            confidence = self.query_model.predict(id_image, query_image)
            total_confidence += confidence

        confidence = total_confidence / total_images
        return confidence.item()

    def _test_data(self):
        test_output = {
            'forgery': [],
            'filename': []
        }
        for file in self.test_dataset.imgs:
            filename = os.path.basename(file[0])
            sig_metadata = SigInfo(filename)
            query_image = Image.open(file[0])
            print('Testing {}'.format(filename))    
            confidence = self._test_id(query_image, sig_metadata.signature)
            print(confidence)
            test_output['forgery'].append('no' if confidence > 0.5 else 'forged')
            test_output['filename'].append(filename)

        self.test_output = test_output
            
    def _write_csv(self, filename):   
        output_df = pd.DataFrame(self.test_output)
        output_df.to_csv(filename, index=False)
        
    def generate_csv(self):
        self._write_csv(self.CSV_FILENAME)


if __name__ == "__main__":
    config, unparsed = get_dir_test_config()
    ID_PATH = config.id_path
    DATA_PATH = config.data_path

    data_tester = Tester(ID_PATH, DATA_PATH)
    print(data_tester.test_output)
    data_tester.generate_csv()