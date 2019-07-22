import theano.tensor as tt
import numpy as np

def cosine_loss(x, y):
    return tt.clip((1 - (x * y).sum(axis=-1) / (norm(x) * norm(y))) / 2, 0, 1)

def norm(x):
    return tt.sqrt(tt.maximum(tt.sqr(x).sum(axis=-1), np.finfo(x.dtype).tiny))


