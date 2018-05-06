import os
import sys
import pickle
import time

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

data = np.load('train-resnet-features.npy').item()
labels = np.concatenate(np.vstack([i.split('/')[1] for i in data.keys()]))
features = np.vstack(list(data.values()))
classes = np.unique(labels)

yTr = torch.tensor(labels)
xTr = torch.tensor(features)

