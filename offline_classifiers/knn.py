"""
kNN
"""

from sklearn.neighbors import KNeighborsClassifier
from aux.BasicClassifierCall import BasicClassifierCall

class Classifier(object):
  def __str__(self):
    return 'kNN'
  
  def models(self):
    # [3, 5, 7, 9, 11, 13, 15, 17, 19]
    return [i for i in xrange(3, 20) if i % 2 != 0] 

  def run(self, model, train, test):
    classifier = KNeighborsClassifier(n_neighbors=model)
    return BasicClassifierCall(classifier, model, train, test)
