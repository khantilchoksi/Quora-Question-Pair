from keras.layers import Input, LSTM, Lambda, Subtract, Dense
from keras.layers import Bidirectional, Flatten, Reshape, dot
from keras.layers import BatchNormalization, Dropout, add
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from keras.optimizers import Adadelta
from keras.utils import Sequence
from keras.models import load_model
from keras.models import Model
from keras import regularizers
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import json
import sys
import os
import io


K.clear_session()


class DataGenerator(Sequence):
    def __init__(self, question_indexes, list_IDs,
                 data_path, name_format, batch_size=128,
                 embedding_len=300, max_len=20, shuffle=True):
        self.question_indexes = question_indexes
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.embedding_len = embedding_len
        self.max_len = max_len
        self.prng = np.random.RandomState(10)
        self.data_path = data_path
        self.name_format = name_format
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def __data_generation(self, list_IDs_temp):
        Left = np.empty((self.batch_size, self.max_len, self.embedding_len))
        Right = np.empty((self.batch_size, self.max_len, self.embedding_len))
        Y = np.empty((self.batch_size, 1))
        for i, ID in enumerate(list_IDs_temp):
            Left[i:i + 1] = np.load(os.path.join(self.data_path, self.name_format.format(self.question_indexes[ID, 0])))
            Right[i:i + 1] = np.load(os.path.join(self.data_path,
                                                  self.name_format.format(self.question_indexes[ID, 1])))
            Y[i] = self.question_indexes[ID, 2:]
        return ([Left, Right], Y)


def attention(sentence_len, embedding_len, units, dropout, name):
    left = Input(shape=(sentence_len, embedding_len), name='{}_Left'.format(name))
    right = Input(shape=(sentence_len, embedding_len), name='{}_Right'.format(name))

    lenc = Bidirectional(LSTM(units, return_sequences=True),
                         merge_mode="sum", name='{}_BD_Left'.format(name))(left)
    renc = Bidirectional(LSTM(units, return_sequences=True),
                         merge_mode="sum", name='{}_BD_Right'.format(name))(right)

    attention = dot([lenc, renc], [1, 1], name='{}_Dot'.format(name))
    attention = Flatten(name='{}_Dot_Flat'.format(name))(attention)
    attention = Dense((1 * units), name='{}_Dense_1'.format(name))(attention)
    attention = Reshape((1, units), name='{}_Reshape'.format(name))(attention)

    merged = add([lenc, attention], name='{}_Add'.format(name))
    merged = Flatten(name='{}_Add_Flat'.format(name))(merged)
    merged = Dense(units, activation='relu', name='{}_Dense_2'.format(name))(merged)
    merged = Dropout(dropout, name='{}_Dropout'.format(name))(merged)
    merged = BatchNormalization(name='{}_BatchNorm'.format(name))(merged)

    is_duplicate = Dense(1, activation='sigmoid', name='{}_Sigmoid'.format(name))(merged)

    model = Model(inputs=[left, right], outputs=is_duplicate, name=name)
    return model


class EpochCallback(Callback):
    def __init__(self, key):
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        d = {}
        if os.path.exists('epochs.json'):
            d = json.load(open('epochs.json'))
        d[self.key] = {}
        d[self.key]['trained'] = 'n'
        d[self.key]['epochs'] = epoch
        json.dump(d, open('epochs.json', 'w'))


def train(indexes, units, dropout, x_train, x_valid, batch_size, name):
    key = name
    training_generator = DataGenerator(indexes, x_train, 'embeddings', '{:06d}.npy', batch_size)
    validation_generator = DataGenerator(indexes, x_valid, 'embeddings', '{:06d}.npy', batch_size)
    embed_x, embed_y = validation_generator.__getitem__(0)

    # optimizer = Adadelta(clipnorm=1.0)

    model = None
    if os.path.exists(os.path.join('models', '{}.h5'.format(key))):
        model = load_model(os.path.join('models', '{}.h5'.format(key)))
        print('Loading model from history...')
    else:
        model = attention(sentence_len=20, embedding_len=300,
                          units=units, dropout=dropout, name=key)
        model.compile('adadelta', 'binary_crossentropy', ['accuracy'])
        print('Created new model...')

    initial_epoch = None
    e = None
    if os.path.exists('epochs.json'):
        e = json.load(open('epochs.json'))
        if key in e:
            initial_epoch = e[key]['epochs']
        else:
            e[key] = {'trained': 'n'}
            initial_epoch = -1
    else:
        e = {key: {'trained': 'n'}}
        initial_epoch = -1

    if not os.path.exists(os.path.join('logs', '{}_log'.format(key))):
        os.mkdir(os.path.join('logs', '{}_log'.format(key)))

    if e[key]['trained'] == 'n':
        epoch_cbk = EpochCallback(key)
        checkpoint = ModelCheckpoint(filepath=os.path.join('models', '{}.h5'.format(key)),
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss',
                                  patience=5,
                                  mode='min')
        tensorboard = TensorBoard(log_dir=os.path.join('logs', '{}_log'.format(key)),
                                  batch_size=batch_size, write_graph=False,
                                  write_grads=False, write_images=False, embeddings_freq=1,
                                  embeddings_layer_names=['{}_Dense_{}'.format(key, x) for x in range(1, 3)])
        csvlogger = CSVLogger(filename=os.path.join('histories', '{}.csv'.format(key)),
                              append=True)
        history = model.fit_generator(generator=training_generator, epochs=100, verbose=1,
                                      callbacks=[epoch_cbk, earlystop, checkpoint, csvlogger, tensorboard],
                                      validation_data=validation_generator,
                                      use_multiprocessing=True, workers=4,
                                      initial_epoch=initial_epoch + 1)
        e = json.load(open('epochs.json'))
        e[key]['trained'] = 'y'
        json.dump(e, open('epochs.json', 'w'))


def main(dropout, units, name):
    indexes = pd.read_csv('indexes.csv').fillna('').values
    X = [i for i in range(indexes.shape[0])]
    X, _, Y, _ = train_test_split(X, indexes[:, 2], test_size=0.1,
                                  random_state=10, shuffle=True, stratify=indexes[:, 2])
    X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10, shuffle=True, stratify=Y)
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('histories'):
        os.mkdir('histories')
    train(indexes, units, dropout, X_train, X_valid, 128, name)


if __name__ == '__main__':
    main(float(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
