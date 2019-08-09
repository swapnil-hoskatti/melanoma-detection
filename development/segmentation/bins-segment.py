import os
import random
from collections import Counter
from copy import copy, deepcopy
from math import sqrt
from statistics import mean as st_mean
from statistics import median
from statistics import mode as st_mode
from statistics import stdev as st_stdev

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import data, feature, filters, io, morphology
from skimage.filters import threshold_adaptive, threshold_otsu
from sklearn.cluster import KMeans


def rsize(img):
	"""
	input: image np array
	return: resized image
	"""
	if img.shape != (450, 600, 3):
		# REF: https://stackoverflow.com/a/48121983/10309266
		img = cv2.resize(img, dsize=(600, 450), interpolation=cv2.INTER_CUBIC)

	return img


def read(path):
	"""
	input: string path of image
	returns: image of fixed size
	"""
	img = cv2.imread(path)

	cv2.imwrite('og.jpg', img)

	return img

	if img:
		return rsize(img)
	else:
		return FileNotFoundError


def __histogram(img):
	"""
	generating hist for R, G, B planes
	returns: (b,g,r) hist dict tuple
			({0:12,...,255:100},{0:12,...,255:100},{0:12,1:33,...,255:100})
	"""

	b_, g_, r_ = cv2.split(img)

	H_b, H_g, H_r = {}, {}, {}

	for pixel in b_.flatten():
		if pixel not in H_b.keys():
			H_b[pixel] = 1
		else:
			H_b[pixel] += 1

	for pixel in g_.flatten():
		if pixel not in H_g.keys():
			H_g[pixel] = 1
		else:
			H_g[pixel] += 1

	for pixel in r_.flatten():
		if pixel not in H_r.keys():
			H_r[pixel] = 1
		else:
			H_r[pixel] += 1

	return H_b, H_g, H_r


def __centerGravity(H_b, H_g, H_r):
	return median(H_b), median(H_g), median(H_r)


def texture(img):
	"""
	Using bins algorithm,
	texture features are obtained and moments are generated

	returns: 8 bins dict
			{000:something, 001:something, 100:something, 010:something,...}
	"""
	bins = {}

	H_b, H_g, H_r = __histogram(img)
	b_median, g_median, r_median = __centerGravity(H_b, H_g, H_r)

	for y, rows in enumerate(img):
		for x, pixel in enumerate(rows):
			for index, value in enumerate(pixel):
				if index == 0:
					# 0 corresponds to B plane in openCV split
					if value > b_median:
						t_b = 1
					else:
						t_b = 0

				if index == 1:
					# 1 corresponds to G plane in openCV split
					if value > g_median:
						t_g = 1
					else:
						t_g = 0

				if index == 2:
					# 2 corresponds to R plane in openCV split
					if value > r_median:
						t_r = 1
					else:
						t_r = 0

			bin_val = f"{t_b} {t_g} {t_r}"

			if bin_val not in bins.keys():
				bins[bin_val] = [(x, y)]
			else:
				bins[bin_val] += [(x, y)]

	return bins


def plotSegment(bins):
	colours = {
		"0 0 0": (0, 69, 255),  # orange
		"0 0 1": (255, 0, 0),  # blue
		"0 1 0": (255, 255, 255),  # white
		"1 0 0": (117, 0, 0),  # navy
		"0 1 1": (0, 0, 128),  # maroon
		"1 1 0": (25, 255, 255),  # yellow
		"1 0 1": (190, 190, 250),  # pink
		"1 1 1": (244, 212, 66)  # cyan
	}

	img = np.zeros((450, 600, 3))

	for bin in colours.keys():
		pixels = False
		try:
			pixels = bins[bin]
		except:
			pass
		if pixels:
			for pixel in pixels:
				img[pixel[1]][pixel[0]] = colours[bin]

	cv2.imwrite('seg.jpg', img)


def remove_hair(img):
	# input img is numpy.ndarray
	src = img

	# Convert the original image to grayscale
	gray_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

	# Kernel for the morphological filtering
	kernel = cv2.getStructuringElement(1, (17, 17))

	# Perform the blackHat filtering on the grayscale image to find the hair contours
	blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

	# intensify the hair contours in preparation for the inpainting algorithm
	ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

	# inpaint the original image depending on the mask
	dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

	# cv2.imwrite(dest_path, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

	cv2.imwrite('rem.jpg', dst)

	return dst

FOLDER = r"/mnt/FOURTH/data/isic/task_3/all/MEL/"
FILE = r"ISIC_0024545.jpg"
PATH = FOLDER + FILE

PATH = r"/mnt/FOURTH/data/isic/task_3/all/hairOG.jpg"
img = read(PATH)
img = remove_hair(img)
bins = texture(img)
plotSegment(bins)
