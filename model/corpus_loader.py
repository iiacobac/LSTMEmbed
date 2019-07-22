import gzip
import numpy as np

def load_corpus(corpus_file, tokenizer, max_dict_size, skip_top_words): 
	texts = []  # list of text samples
	f = open("data/text8")
	texts.append(f.read())
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	ret = []
	for item in sequences[0]:
		# infrequent words and stopwords filter
		if item >= max_dict_size or item < skip_top_words:
			ret.append(0)
		else:
			ret.append(item)
	return ret

#print(sequence2)

if __name__ == '__main__':
	fil = filters='!"#$%&()*+,-./;<=>?@[\\]^`{|}~\t\n'
	tokenizer = Tokenizer(num_words=500000000, filters=fil)
	corpus = load_corpus('data/text8', tokenizer, 500000, 100)
	print corpus[0]



