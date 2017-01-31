from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import matplotlib.pyplot as plt


def maybe_download(filename, expected_bytes, url):
	"""Download a file if not present, and make sure it's the right size."""
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception(
				'Failed to verify ' + filename + '. Can you get to it with a browser?')
	return filename

def read_data(filename):
	"""Extract the first file enclosed in a zip file as a list of words"""
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

def calculate_NEG_sub_distribution(word_cnt, BOW_size, t_sub):
	""" Calculate the NEG distribution Pn(w) ~ U(w)**3/4 and subsampling prabablity P(w)"""
	total = 0
	total_p = 0
	Pw_dict = {}
	P_n = np.zeros(BOW_size)				# U(w)**3/4 for negative sampling
	for i, (_, num) in enumerate(word_cnt):
		total += num
		total_p += num**(3/4)
		P_n[i] = num**(3/4)
	P_n = P_n / total_p
	
	for word, num in word_cnt:
		Pw_dict[word] = 1 - (t_sub/(num/total))**0.5

	return P_n, Pw_dict

def choose_NEG_sample(P_n):
	prob = random.random()
	for index, p in enumerate(P_n):
		prob -= p
		if prob < 0:
			return index
	if prob > 0:
		raise NameError("P_n isn't valid pdf")

	
batch_index = 0
def next_batch(batch_size, train_set, BOW_size):
	"""Return a batch_size batch tuple from input train_set"""
	global batch_index
	batch_tuple = []
	if (batch_index + batch_size) > len(train_set):
		batch_tuple = train_set[batch_index:]
		batch_index = batch_size - (len(train_set) - batch_index)
		batch_tuple += train_set[:batch_index]
	else:
		for i in range(batch_index, batch_index + batch_size):
			batch_tuple.append((train_set[i][0], train_set[i][1]))
			batch_index += 1
	
	x = np.zeros(shape=(batch_size, BOW_size), dtype=np.float32)
	y_ = np.zeros(shape=(batch_size, BOW_size), dtype=np.float32)
	
	for i in range(batch_size):
		x[i][batch_tuple[i][0]] = 1
		y_[i][batch_tuple[i][1]] = 1
	
	return x, y_

def next_batch_in_index(batch_size, train_set, BOW_size):
	"""Return a batch_size index array from input train_set"""
	global batch_index
	batch_tuple = []
	if (batch_index + batch_size) > len(train_set):
		batch_tuple = train_set[batch_index:]
		batch_index = batch_size - (len(train_set) - batch_index)
		batch_tuple += train_set[:batch_index]
	else:
		for i in range(batch_index, batch_index + batch_size):
			batch_tuple.append((train_set[i][0], train_set[i][1]))
			batch_index += 1
	
	x = np.zeros(shape=(batch_size), dtype=np.int32)
	y_ = np.zeros(shape=(batch_size), dtype=np.int32)
	
	for i in range(batch_size):
		x[i] = batch_tuple[i][0]
		y_[i] = batch_tuple[i][1]
	
	return x, y_


def plot_with_labels(low_dim_embs, labels, filename='word2vec_tsne.png'):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	plt.figure(figsize=(30, 30))	# in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label,
								 xy=(x, y),
								 xytext=(5, 2),
								 textcoords='offset points',
								 ha='right',
								 va='bottom')

	plt.savefig(filename)
	plt.close()	
	
	