"""
Gaussian NB
"""

from sklearn.naive_bayes import GaussianNB
from aux.BasicClassifierCall import BasicClassifierCall

class Classifier(object):
  def __str__(self):
    return 'Gaussian NB'
  
  def models(self):
    return ['N/A']

  def run(self, model, train, test):
    classifier = GaussianNB()
    return BasicClassifierCall(classifier, model, train, test)

