import os
from sklearn import preprocessing
from itertools import product
import pandas as pd
from sklearn.metrics import f1_score
import warnings

def BasicClassifierCall(classifier, model, train, test):
    X_train, y_train = train
    X_test, y_test = test

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      macro = f1_score(y_test, y_pred, average='macro') 
      micro = f1_score(y_test, y_pred, average='micro') 

    return [macro, micro]
