import numpy as np
import random
from six.moves import range
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
import re

def tokenize(sequence, word_index, dict_size=3000000, window_size=1, skip=100, embeddings_index={}):
	input1 = []
	input2 = []
	outputs = []
	for i, wi in enumerate(sequence):
		if not wi or wi < skip or wi >= dict_size or (len(embeddings_index) > 0 and wi not in embeddings_index):
			continue
		arr_pre = []
		arr_post = []
		ok_pre = False
		ok_post = False
		window_start = max(0, i-window_size)
		window_end = min(len(sequence), i+window_size+1)
		for j in range(i-window_size, i):
			if j < window_start:
				arr_pre.append(0)
				continue
			wj = sequence[j]
			if not wj:
				arr_pre.append(0)
				continue
			arr_pre.append(wj)
		for j in reversed(range(i+1,i+window_size+1)):
			if j >= window_end:
				arr_post.append(0)
				continue
			wj = sequence[j]
			if not wj:
				arr_post.append(0)
				continue
			arr_post.append(wj)
		input1.append(arr_pre)
		input2.append(arr_post)
		outputs.append(wi)
	inputs = [np.array(input1), np.array(input2)]
	return inputs, np.array(outputs)


