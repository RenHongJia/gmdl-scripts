"""
kNN
"""

from sklearn.neighbors import KNeighborsClassifier
from aux.BasicClassifierCall import BasicClassifierCall

class Classifier(object):
  def __str__(self):
    return 'kNN'
  
  def models(self):
    return [3, 5, 7, 9, 11]

  def run(self, model, train, test):
    classifier = KNeighborsClassifier(n_neighbors=model)
    return BasicClassifierCall(classifier, model, train, test)
