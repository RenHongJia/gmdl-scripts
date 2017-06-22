"""
SVM
"""

from sklearn.svm import SVC
from aux.BasicClassifierCall import BasicClassifierCall
from itertools import product

class Classifier(object):
  def __init__(self):
    print 'SVM'
  
  def models(self):
    C = [0.5, 1, 3]
    kernel = ['rbf', 'linear']
    gamma = [0.1, 1, 'auto']

    combinations = list(product(kernel, gamma, C))

    combinations = map(
      lambda x: x if x[0] is 'rbf' else (x[0], 'auto', x[2]), 
      combinations
    )

    return list(set(combinations))

  def run(self, model, path, train, test, set_name, id=0):
    classifier = SVC(C=model[2], kernel=model[0], gamma=model[1])
    return BasicClassifierCall(classifier, model, path, train, test)

