"""
GMDL

requires GMDL_PATH as an ENV variable
"""

from aux.GMDL import GMDL
from aux.BasicClassifierCall import BasicClassifierCall
from itertools import product

class Classifier(object):
  def __str__(self):
    return 'GMDL'
  
  def models(self):
    sigma = [0.5, 1, 2, 3]
    omega = [8, 16, 32, 64]

    return list(product(sigma, omega))

  def run(self, model, train, test):
    classifier = GMDL(sigma=model[0], omega=model[1])
    return BasicClassifierCall(classifier, model, train, test)
