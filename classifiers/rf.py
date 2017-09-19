"""
Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
from aux.BasicClassifierCall import BasicClassifierCall
from itertools import product

class Classifier(object):
  def __str__(self):
    return 'Random Forest'
  
  def models(self):
    # [10, 20, 30, 40, 50, 60, 70, 80, 90]
    trees = list(xrange(10, 100, 10))
    criterion = ['gini', 'entropy']

    return list(product(trees, criterion))

  def run(self, model, train, test):
    classifier = RandomForestClassifier(n_estimators=model[0], criterion=model[1])
    return BasicClassifierCall(classifier, model, train, test)

