# Verifikator

Offline Writer Independent Signature Verification Model and REST API

# Setup Environment
    - Create a Python 3.6 virtual environment or a conda environment (preferred).
        `python3 -m venv env`
    - Activate the environment.
        `source ./env/bin/activate on Linux`
        `./env/Scripts/activate.bat on Windows`
        `activate <env-name> for conda envs on Windows`
    - Install the required libraries
        `pip3 install -r requirements.txt` or `conda install -r requirements.txt` 
    - Training the model
        - `./model`
            - Copy the dataset to `./dataset/signatures` (it should contain `train` and `valid` folders inside.)
            - example training command
                `python main.py --num_model 1 --batch_size 256`
    - Using the API.
        - `./app`
            - Copy the model to `./model/model_ckpt.tar`
            - `python application.py` or `python application.py --use_gpu`

    - Play
        - Play with the model using the provided `test_notebook.ipynb`
# References

- [https://github.com/kevinzakka/one-shot-siamese](https://github.com/kevinzakka/one-shot-siamese)
- [https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/abs/1707.02131 )