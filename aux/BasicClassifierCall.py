import os
from sklearn import preprocessing
from itertools import product
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import warnings

def BasicClassifierCall(classifier, model, train, test, labels):
    X_train, y_train = train
    X_test, y_test = test

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    classifier.fit(X_train, y_train)
    y_pred = pd.Series(classifier.predict(X_test), dtype=y_test.dtype)

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      classes = unique_labels(y_test, y_pred)
      cm = confusion_matrix(y_test, y_pred, labels=labels)

    return pd.DataFrame(cm, columns=classes)
