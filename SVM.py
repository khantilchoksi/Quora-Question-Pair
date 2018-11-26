
# coding: utf-8

# # SVM

# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pickle
import sys
import os


# ## Load the Data

# In[ ]:


train = pd.read_csv('pca_train_2.csv').fillna('').values
test = pd.read_csv('pca_test_2.csv').fillna('').values


# ## Split into X and Y

# In[ ]:


X_train, Y_train = train[:,:-1], train[:,-1]
X_test, Y_test = test[:,:-1], test[:,-1]


# ## SVM Cross-Validation

# In[ ]:


clf = None
if not os.path.exists('svm.pkl'):
    param_grid = [
        {'C':[0.001, 0.01, 0.1, 1, 10, 100],
         'kernel':['linear', 'sigmoid', 'rbf', 'poly'],
         'degree':[2,3],
         'gamma': [0.1,0.3,0.5,0.7,0.9],
         'coef0': [0.1,0.5,1.0,5.0,10.0]}
    ]

    svc = SVC(random_state=10)
    clf = GridSearchCV(svc, param_grid,
                       ['accuracy'],
                       cv=5, refit='accuracy', verbose=1, n_jobs=4)
    clf.fit(X_train, Y_train)
else:
    clf = pickle.load(open('svm.pkl', 'rb'))


# ## Save SVM Model

# In[ ]:


if not os.path.exists('svm.pkl'):
    pickle.dump(clf, open('svm.pkl', 'wb'))


# ## Predict and Output Results

# In[ ]:


Y_pred = clf.predict(pca.transform(X_test))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

