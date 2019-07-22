from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, merge
from tools.utils import cosine_loss

def create_lstmembed_model(vocab_size, input_length=10, nhidden=200, emb_dimension=400):
	'''
	 'nhidden':100, # number of hidden units
	 'emb_dimension':100, # dimension of word embedding
	 'dropout': # percentaje dropout
	'''
	pre_context = Input(shape=(input_length,))
	post_context = Input(shape=(input_length,))
	emb_layer = Embedding(output_dim=nhidden, input_dim=vocab_size, input_length=input_length)
	pre_embed = emb_layer(pre_context)
	post_embed = emb_layer(post_context)	
	pre_lstm = LSTM(nhidden, consume_less='gpu')(pre_embed)
	post_lstm = LSTM(nhidden, consume_less='gpu')(post_embed)
	merged_vector = merge([pre_lstm, post_lstm], mode='concat')
	predictions = Dense(emb_dimension, activation='linear')(merged_vector)
	model = Model(input=[pre_context, post_context], output=predictions)
	model.compile(loss=cosine_loss, 
		optimizer='adam', 
		metrics=['accuracy', 'cosine_proximity'])

	return model

if __name__ == '__main__':
	model = create_lstmembed_model(10, 10, 10, 10)
	print(model.summary())
