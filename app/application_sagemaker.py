"""

Verifikator API
    - https://github.com/arush15june/Verifikator
"""

import sys
import time
from io import BytesIO

from flask import Flask, request, make_response, jsonify, Response#, send_from_directory
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
from werkzeug.datastructures import FileStorage

from helpers import ModelPredictor

DEFAULT_MODEL_PATH = "../model/model_ckpt.tar"
DEFAULT_DATASET_DIR = "../model/dataset/"
DEFAULT_IMAGES_DIR = DEFAULT_DATASET_DIR+"signatures/valid/"

"""
    Flask Config
"""

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

app = Flask(__name__)
app.config.from_object(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

api = Api(app)

class FileStorageArgument(reqparse.Argument):
    """This argument class for flask-restful will be used in
    all cases where file uploads need to be handled."""
    
    def convert(self, value, op):
        if self.type is FileStorage:  # only in the case of files
            # this is done as self.type(value) makes the name attribute of the
            # FileStorage object same as argument name and value is a FileStorage
            # object itself anyways
            return value

        # called so that this argument class will also be useful in
        # cases when argument type is not a file.
        super(FileStorageArgument, self).convert(*args, **kwargs)


class SigVerify(Resource):
    image_parser = reqparse.RequestParser(argument_class=FileStorageArgument)
    image_parser.add_argument('image', required=True, type=FileStorage, location='files')

    req_parser = reqparse.RequestParser()
    req_parser.add_argument('cust_id', required=True, location='args')

    predictor = ModelPredictor(DEFAULT_MODEL_PATH, DEFAULT_IMAGES_DIR)

    @staticmethod
    def verify_extension(image):

        extension = image.filename.rsplit('.', 1)[1].lower()
        if '.' in image.filename and not extension in app.config['ALLOWED_EXTENSIONS']:
            return False
        else:
            return True

    def post(self, cust_id):
        image_args = self.image_parser.parse_args()
        # req_args = self.req_parser.parse_args()
        
        image = image_args['image']

        if not self.verify_extension(image):
            abort(400, message='Unsupported File Extension')

        image_file = BytesIO()
        try:
            image.save(image_file)
        except:
            abort(400, message="Invalid Input")
        start = time.time()
        confidence, is_genuine = self.predictor.getConfidence(image_file, cust_id)

        app.logger.info('/api/verify: {} s taken to generate response'.format(time.time() - start))
        
        data_dict = {
            'confidence': confidence.item(),
            'genuine': is_genuine
        }

        return make_response(jsonify(data_dict))

api.add_resource(SigVerify, "/invocations")
api.add_resource(Ping, "/ping")

if __name__ == '__main__':
    app.run(debug=True)