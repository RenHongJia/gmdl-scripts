"""
SVM
"""

from sklearn.svm import LinearSVC
from aux.BasicClassifierCall import BasicClassifierCall
from itertools import product

class Classifier(object):
  def __str__(self):
    return 'SVM'
  
  def models(self):
    # [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    C = [10 ** i for i in xrange(-5, 5)]
    loss = ['hinge', 'squared_hinge']
    penalty = ['l1', 'l2']

    return list(product(C, loss, penalty))

  def run(self, model, train, test):
    classifier = LinearSVC(C=model[0], loss=model[1], penalty=model[2])
    return BasicClassifierCall(classifier, model, train, test)

