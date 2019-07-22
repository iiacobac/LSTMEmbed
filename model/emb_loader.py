import gzip
import numpy as np

def load_embeddings(emb_file, word_index, dict_size): 
	embeddings_index = {}
	with gzip.open(emb_file, 'rb') as f: 
		for line in f:
			values = line.split()
			word = values[0]
			if word in word_index:
				if word_index[word] < dict_size:
					coefs = np.asarray(values[1:], dtype='float32')
					embeddings_index[word_index[word]] = coefs
	return embeddings_index

if __name__ == '__main__':
	index = load_embeddings('data/emb.txt.gz', {'bank':1, 'bar':2}, 3)
	print index


