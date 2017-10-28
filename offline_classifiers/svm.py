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
    # C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    return [10 ** i for i in xrange(-4, 4)]

  def run(self, model, train, test):
    classifier = LinearSVC(C=model)
    return BasicClassifierCall(classifier, model, train, test)

