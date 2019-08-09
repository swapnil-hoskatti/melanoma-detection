# imports - standard imports
import os
import random
from collections import Counter
from copy import copy, deepcopy
from math import sqrt
from statistics import mean as st_mean
from statistics import median
from statistics import mode as st_mode
from statistics import stdev as st_stdev

# imports - third party imports
import cv2
import keras
import numpy as np
from scipy import ndimage, stats
from skimage import data, feature, filters, io, morphology
from skimage.filters import threshold_adaptive, threshold_otsu
from sklearn.cluster import KMeans

# global cnn table : extracted from Keras settings
cnn_dict = {
	1 : 'Melanoma',
	0 : 'Nevus'
}