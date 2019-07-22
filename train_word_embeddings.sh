make
if [ ! -e data/text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O data/text8.gz
  gzip -d data/text8.gz -f
fi
## data/emb.txt.gz is missing by anonymyty reasons 
## just a plain embeddings file.
python model/runner.py data/text8 data/emb.txt.gz emb.vec

