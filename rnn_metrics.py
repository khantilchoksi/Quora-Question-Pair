from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from keras.utils import Sequence
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import pickle
import sys
import os


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


def main(lstm_model, attention_model):
    results = pickle.load(open('results.pkl', 'rb'))
    metrics = pd.read_csv('metrics.csv')

    indexes = pd.read_csv('indexes.csv').fillna('').values
    X = [i for i in range(indexes.shape[0])]
    X_train, X_test, Y_train, Y_test = train_test_split(X, indexes[:, 2], test_size=0.1,
                                                        random_state=10, shuffle=True, stratify=indexes[:, 2])
    X_train, X_valid = train_test_split(X_train, test_size=0.1,
                                        random_state=10, shuffle=True, stratify=Y_train)

    training_generator = DataGenerator(indexes, X_train, 'embeddings', '{:06d}.npy', 2599)
    validation_generator = DataGenerator(indexes, X_valid, 'embeddings', '{:06d}.npy', 311)
    testing_generator = DataGenerator(indexes, X_test[:-1], 'embeddings', '{:06d}.npy', 36, shuffle=False)

    K.clear_session()
    lstm = load_model(os.path.join('models', lstm_model))
    scores = lstm.predict_generator(testing_generator, verbose=1).ravel()
    # for i in range(len(scores)):
    #     print(Y_test[i], scores[i])
    fpr, tpr, thresholds = roc_curve(Y_test[:-1], scores, pos_label=1)
    test_acc = lstm.evaluate_generator(testing_generator)[1]
    train_acc = lstm.evaluate_generator(training_generator)[1]
    val_acc = lstm.evaluate_generator(validation_generator)[1]
    name = 'LSTM'
    results.append((fpr, tpr, name))
    metrics.loc[len(metrics)] = [name, test_acc, train_acc, val_acc]

    K.clear_session()
    attention = load_model(os.path.join('models', attention_model))
    scores = attention.predict_generator(testing_generator, verbose=1).ravel()
    fpr, tpr, thresholds = roc_curve(Y_test[:-1], scores, pos_label=1)
    test_acc = attention.evaluate_generator(testing_generator)[1]
    train_acc = attention.evaluate_generator(training_generator)[1]
    val_acc = attention.evaluate_generator(validation_generator)[1]
    name = 'Attention'
    results.append((fpr, tpr, name))
    metrics.loc[len(metrics)] = [name, test_acc, train_acc, val_acc]

    pickle.dump(results, open('results.pkl','wb'))
    metrics.to_csv('metrics.csv')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
