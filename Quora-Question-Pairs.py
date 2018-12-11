import numpy as np
import pandas as pd
import nltk
import re
import os
import pickle
import time
import datetime
from collections import defaultdict, Counter
from string import punctuation
import numpy.random as rn

# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from hypopt import GridSearch

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier

import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, model_from_json, load_model
from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Conv1D, Dense, Activation
from keras.layers import Dropout, BatchNormalization, Flatten, Reshape, Lambda, dot, add
from keras.regularizers import l2
from keras.optimizers import Adadelta, SGD
from keras.activations import selu
from keras.callbacks import Callback, ModelCheckpoint
from keras.constraints import maxnorm

np.random.seed(1234)

WORD_EMBEDDING_FILE = "fasttext.pkl"
EMBEDDING_LEN = 300

with open(WORD_EMBEDDING_FILE, 'rb') as input:
    word_embeddings = pickle.load(input)
print("Embeddings loaded.")

data = pd.read_csv('cleaned-data.csv')
print("Size of the dataset: ", data.shape)
print("Columns: ", data.columns)

data[data.isnull().T.any().T]
d = data.dropna()
del data
df = d.sample(frac=0.5, replace=False)
del d

q1 = "question1"
q2 = "question2"
is_duplicate = 'is_duplicate'
df = df[[q1, q2, is_duplicate]]
print("Subset selected.")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df[q1].tolist() + df[q2].tolist())

# Replaces every word in the question with the index of the word in the dictionary.
df[q1] = tokenizer.texts_to_sequences(df[q1])
df[q2] = tokenizer.texts_to_sequences(df[q2])

# Dictionary of (index,word) pairs
vocabulary = tokenizer.word_index

# Initializing embeddings matrix with random values
embeddings = 1 * np.random.randn(len(vocabulary) + 1, EMBEDDING_LEN)
embeddings[0] = 0  # Embedding vector for unrecognized words

# Filling up the embedding matrix with actual embedding values
for word, index in vocabulary.items():
    if word in word_embeddings:
        embeddings[index] = word_embeddings[word]

print("Embeddings generated.")

results = []
metrics = pd.DataFrame(columns=['test_accuracy', 'train_accuracy', 'val_accuracy'])


def get_Sentence_Vector(words):
    # Input: Sentence as a list of word indices
    # Returns the average of word embeddings of all words in the string
    total = np.zeros(300)
    for index in words:
        total += embeddings[index]
    count = len(words)
    return total / count


X1 = df[q1].apply(lambda x: get_Sentence_Vector(x))
X2 = df[q2].apply(lambda x: get_Sentence_Vector(x))

new1 = pd.DataFrame(X1.values.tolist(), index=X1.index)
new2 = pd.DataFrame(X2.values.tolist(), index=X1.index)

new = new1.join(new2, lsuffix="q1_", rsuffix="q2_")

x1 = np.array(X1.values.tolist())
x2 = np.array(X2.values.tolist())

# Generating additional features by multiplying the average vectors of two sentences
X1_X2 = pd.DataFrame(x1 * x2, index=X1.index)

new = new.join(X1_X2, rsuffix="q1q2_")
new = new.fillna(0)
print(new.shape)
print("Merged Data created.")

del X1
del X2
del new1
del new2
del x1
del x2
del X1_X2

X_train, X_test, Y_train, Y_test = train_test_split(new, df[is_duplicate], test_size=0.06, stratify=df[is_duplicate])
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.06, stratify=Y_train)

print("Train dataset size: ", X_train.shape[0])
print("Validation dataset size: ", X_val.shape[0])
print("Test dataset size: ", X_test.shape[0])

del new


# ### Logistic Regression
parameters = [{"penalty": ["l1", "l2"], 'C': [0, 3, 0.5, 0.6]}]

clf = LogisticRegression()
if not os.path.exists('final_logistic.pkl'):
    clf = GridSearch(model=clf)
    clf.fit(X_train, Y_train, parameters, X_val, Y_val)
else:
    clf = pickle.load(open('final_logistic.pkl', 'rb'))
print(clf)

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:, 1]

test_acc = clf.score(X_test, Y_test)
train_acc = clf.score(X_train, Y_train)
val_acc = clf.score(X_val, Y_val)

fpr, tpr, thresholds = roc_curve(Y_test, scores, pos_label=1)

name = "Logistic Regression"
results.append((fpr, tpr, name))
metrics.loc[name] = np.array([test_acc, train_acc, val_acc])
pickle.dump(clf, open('final_logistic.pkl', 'wb'))
del clf


# ### Gaussian NaiveBayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
if not os.path.exists('final_gaussian.pkl'):
    clf.fit(X_train, Y_train)
else:
    clf = pickle.load(open('final_gaussian.pkl', 'rb'))

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:, 1]

test_acc = clf.score(X_test, Y_test)
train_acc = clf.score(X_train, Y_train)
val_acc = clf.score(X_val, Y_val)

fpr, tpr, thresholds = roc_curve(Y_test, scores, pos_label=1)

name = "Gaussian NaiveBayes"
results.append((fpr, tpr, name))
metrics.loc[name] = np.array([test_acc, train_acc, val_acc])
pickle.dump(clf, open('final_gaussian.pkl', 'wb'))
del clf


# ### Random Forest
parameters = {'max_depth': [4, 5, 6], 'n_estimators': [15, 20, 25]}
clf = RandomForestClassifier()
if not os.path.exists('final_random_forest.pkl'):
    clf = GridSearch(model=clf)
    clf.fit(X_train, Y_train, parameters, X_val, Y_val)
else:
    clf = pickle.load(open('final_random_forest.pkl', 'rb'))
print(clf)

pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:, 1]

test_acc = clf.score(X_test, Y_test)
train_acc = clf.score(X_train, Y_train)
val_acc = clf.score(X_val, Y_val)

fpr, tpr, thresholds = roc_curve(Y_test, scores, pos_label=1)

name = "RandomForest"
results.append((fpr, tpr, name))
metrics.loc[name] = np.array([test_acc, train_acc, val_acc])
pickle.dump(clf, open('final_random_forest.pkl', 'wb'))
del clf


# ### Feed forward Neural network
def feed_forward_model():
    model = Sequential()
    model.add(Dense(60, activation="relu", kernel_initializer="uniform", input_dim=900))
    model.add(Dropout(0.55))
    model.add(Dense(1, activation="sigmoid"))
    return model


from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
EPOCHS = 100
BATCH_SIZE = 128

model = feed_forward_model()

# optimizer=Adadelta(clipnorm=1.5)
if not os.path.exists('final_feed_forward.h5'):
    optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    checkpointer = ModelCheckpoint(filepath='final_feed_forward.h5',
                                   monitor='val_loss', verbose=1,
                                   save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    csvlogger = CSVLogger(filename='feed_forward.csv')
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        callbacks=[checkpointer, earlystopping, csvlogger])
else:
    model = load_model('final_feed_forward.h5')

scores = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(Y_test, scores, pos_label=1)

test_acc = model.evaluate(X_test, Y_test)[1]
train_acc = model.evaluate(X_train, Y_train)[1]
val_acc = model.evaluate(X_val, Y_val)[1]

name = "Feed Forward Neural Network"
results.append((fpr, tpr, name))
metrics.loc[name] = np.array([test_acc, train_acc, val_acc])

pickle.dump(results, open('results.pkl', 'wb'))
metrics.to_csv('metrics.csv')
