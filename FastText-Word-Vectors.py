from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
import json
import sys
import os
import io


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(x) for x in tokens[1:]]
    return data


embeddings = load_vectors('wiki.simple.vec')
data = pd.read_csv('questions.csv').fillna("NA")
question_data = data.values
MAX_LEN = 20
EMB_LEN = 300


nullembedding = np.random.randn(1, MAX_LEN, EMB_LEN)
np.save(os.path.join('embeddings','{:06d}'.format(0)), nullembedding)


out_of_vocab = []
for i in range(question_data.shape[0]):
    words = question_data[i,1].split(' ')
    for word in words:
        if word not in embeddings:
            out_of_vocab.append(word)


out_of_vocab = set(out_of_vocab)
with open('queries.txt', 'w') as fp:
    for word in out_of_vocab:
        fp.write('{}\n'.format(word))

# do fasttext out-of-vocab generation here.
oov_embeddings = load_vectors('qq.oov.vec')


for i in range(question_data.shape[0]):
    x = pad_sequences([[embeddings[word] if word in embeddings else oov_embeddings[word] for word in question_data[i,1].split(' ')]],
                      maxlen=MAX_LEN, dtype='float32', padding='pre',
                      truncating='pre', value=0.0)
    np.save(os.path.join('embeddings','{:06d}'.format(question_data[i,0])), x)
    if i%100==0:
        print('{}'.format(i))
    del x
