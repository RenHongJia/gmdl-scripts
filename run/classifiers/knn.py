"""
kNN
"""

from sklearn.neighbors import KNeighborsClassifier
from aux.BasicClassifierCall import BasicClassifierCall

class Classifier(object):
  def __init__(self):
    print 'kNN'
  
  def models(self):
    return [3, 5, 7, 9, 11]

  def run(self, model, path, train, test, set_name, id=0):
    classifier = KNeighborsClassifier(n_neighbors=model)
    return BasicClassifierCall(classifier, model, path, train, test)
