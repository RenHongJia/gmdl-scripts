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
    trees = [5, 10, 20, 30]
    criterion = ['gini', 'entropy']

    return list(product(trees, criterion))

  def run(self, model, path, train, test, set_name, id=0):
    classifier = RandomForestClassifier(n_estimators=model[0], criterion=model[1])
    return BasicClassifierCall(classifier, model, path, train, test)

