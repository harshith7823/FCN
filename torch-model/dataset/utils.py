import os
from glob import glob
import re
import torch.nn as nn
from torch.autograd import Variable
import torch
import cv2
import json
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt

np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(img, mean, std):
    img = img/255.0
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    img = np.clip(img, 0.0, 1.0)
    return img

def get_label_paths(label_path):
    label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                   for path in glob(os.path.join(label_path, '*_road_*.png'))}
    return label_paths

def get_test_paths(test_path):
    test_paths = [os.path.basename(path)
                      for path in glob(os.path.join(test_path, '*.png'))]

    return test_paths

def resize_label(image_path, label):
    image = io.imread(image_path)
    label = transform.resize(label, image.shape)
    output = cv2.addWeighted(image, 0.6, label, 0.4, 0, dtype = 0)
    return output
