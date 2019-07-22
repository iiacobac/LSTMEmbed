from emb_loader import load_embeddings
from corpus_loader import load_corpus
from lstm_model import create_lstmembed_model
from text_to_sequences import tokenize
from keras.preprocessing.text import Tokenizer 

import numpy as np

import sys

def runner(corpus_file, emb_file, out_file):
	s = {'nhidden':50, 			# number of hidden units
        'emb_dimension':50, 	# dimension of word embedding
        'nepochs':4,
		'batch_size':2048,
		'window_size':10,
		'skip_top_words':100,
	 	'max_vocab_size':50000}

	# Tokenizer 
	fil = filters='!"#$%&()*+,-./;<=>?@[\\]^`{|}~\t\n'
	tokenizer = Tokenizer(num_words=1000000000, filters=fil)

	# loading corpus
	sequence = load_corpus(corpus_file, tokenizer, s['max_vocab_size'], s['skip_top_words'])

	# loading embeddings
	embeddings_index = load_embeddings(emb_file, tokenizer.word_index, s['max_vocab_size'])

	# text to sequences
	inputs, outputs = tokenize(sequence, tokenizer.word_index, window_size=s['window_size'], skip=s['skip_top_words'], embeddings_index=embeddings_index)

	out = []
	for index in outputs:
		if index > 0 and index in embeddings_index:
			out.append(embeddings_index[index])

	y = np.array(out)

	vocab_size = min(s['max_vocab_size'], len(tokenizer.word_index) + 1)

	model = create_lstmembed_model(vocab_size, s['window_size'], s['nhidden'], s['emb_dimension'])

	model.fit(inputs, y, epochs=s['nepochs'], batch_size=s['batch_size'])
	
	lookup = {tokenizer.word_index[key]: key for key in tokenizer.word_index}	
	vec = model.get_weights()[0]
	with open(out_file,'w') as f: 
		for index in range(s['skip_top_words'], s['max_vocab_size']):
			if index in lookup:
				f.write(lookup[index])
				f.write(" ")
				f.write(" ".join(map(str, list(vec[index,:]))))
				f.write("\n")


if __name__ == '__main__':
	runner(sys.argv[1], sys.argv[2], sys.argv[3])



