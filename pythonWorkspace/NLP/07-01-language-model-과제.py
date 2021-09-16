# !pip install sentencepiece

import os
import random
import shutil
import json
import zipfile
import math
import copy
import collections
import re

import matplotlib.pyplot as plt #pip install matplotlib
import pandas as pd #pip install pandas
import numpy as np
import sentencepiece as spm
import tensorflow as tf #pip install tensorflow
import tensorflow.keras.backend as K 

from tqdm.notebook import tqdm #pip install tqdm


# !nvidia-smi


# random seed initialize

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


# # google drive mount
# from google.colab import drive #pip install google.colab
# drive.mount('/content/drive')


# # data dir
# data_dir = '/content/drive/MyDrive/Data/nlp'
# os.listdir(data_dir)


# # korean wiki dir
# kowiki_dir = os.path.join(data_dir, 'kowiki')
# if not os.path.exists(kowiki_dir):
#     os.makedirs(kowiki_dir)
# os.listdir(kowiki_dir)


# vocab loading
vocab = spm.SentencePieceProcessor()
vocab = load_model('ko_32000.model')


# n_vocab = len(vocab)  # numbezr of vocabulary
# n_seq = 256  # number of sequence
# d_model = 256  # dimension of model


