# LSTMEmbed
[Learning Word and Sense Representations from a Large Semantically Annotated Corpus with Long Short-Term Memories](https://www.aclweb.org/anthology/P19-1165)

https://github.com/iiacobac/LSTMEmbed

## Setup

Download a word-based embeddings file like from [wor2vec](https://code.google.com/archive/p/word2vec/) or [Glove](https://nlp.stanford.edu/projects/glove/) or a sense-based file like from [SensEmbed](https://iiacobac.wixsite.com/sensembed), and place it in data/


## Training

python train_word_embeddings.sh

## Requirements

   Python 2.7
   Keras 2
   
## Trained Word and Sense Embeddings

Follow this [link](https://drive.google.com/open?id=1vjhWpR-VIxNj_9ED_bH6LtwxDs2CY8XG)

## Reference

Main paper to be cited

	@inproceedings{iacobacci-navigli-2019-lstmembed,
    		title = "{LSTME}mbed: Learning Word and Sense Representations from a Large Semantically Annotated Corpus with Long Short-Term Memories",
    		author = "Iacobacci, Ignacio and Navigli, Roberto",
		booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    		month = jul,
    		year = "2019",
    		address = "Florence, Italy",
    		publisher = "Association for Computational Linguistics",
    		url = "https://www.aclweb.org/anthology/P19-1165",
    		pages = "1685--1695",
    		abstract = "While word embeddings are now a de facto standard representation of words in most NLP tasks, recently the attention has been shifting towards vector representations which capture the different meanings, i.e., senses, of words. In this paper we explore the capabilities of a bidirectional LSTM model to learn representations of word senses from semantically annotated corpora. We show that the utilization of an architecture that is aware of word order, like an LSTM, enables us to create better representations. We assess our proposed model on various standard benchmarks for evaluating semantic representations, reaching state-of-the-art performance on the SemEval-2014 word-to-sense similarity task. We release the code and the resulting word and sense embeddings at http://lcl.uniroma1.it/LSTMEmbed.",
	}

============================================

## Support

For more information, bug reports, fixes, please contact:  

Ignacio Iacobacci   
iiacobac[at]gmail[dot]com  
http://iiacobac.wordpress.com/  

Roberto Navigli   
navigli[at]di[dot]uniroma1[dot]it   
http://wwwusers.di.uniroma1.it/~navigli/

## License

LSTMEmbed is an output of the MOUSSE ERC Consolidator Grant  No. 726487.
LSTMEmbed authors gratefully acknowledge the support of NVIDIA Corporation Hardware Grant.
LSTMEmbed is licensed under a [Creative Commons Attribution - Noncommercial - Share Alike 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/) License.
