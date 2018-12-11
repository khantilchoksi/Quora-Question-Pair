
# coding: utf-8

# # Cleaned Data

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import pickle
import json
import sys
import os
import io


# ## Load Embedding Function

# In[ ]:


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(x) for x in tokens[1:]]
    return data


# ## Load All Embeddings

# In[ ]:


fasttext = load_vectors('wiki.simple.vec')
outofvocab = load_vectors('qq.oov.vec')
embeddings = fasttext.copy()
embeddings.update(outofvocab)


# In[ ]:


import pickle
pickle.dump(embeddings, open('fasttext.pkl', 'wb'))


# In[ ]:


del fasttext
del outofvocab


# ## Load the Data Indexes

# In[ ]:


indexes = pd.read_csv('indexes.csv').fillna('')
npindexes = indexes.values


# ## Load the Questions

# In[ ]:


questions = pd.read_csv('questions.csv').fillna('NA').values


# ## Join the Indexes and Questions

# In[ ]:


indexes['question1'] = questions[npindexes[:,0]-1,1]


# In[ ]:


indexes['question2'] = questions[npindexes[:,1]-1,1]


# In[ ]:


indexes.to_csv('cleaned-data.csv', index=False)

