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
    sigma = [1, 5, 10]
    tau = [0]
    return list(product(sigma, tau))

  def run(self, model, train, test):
    classifier = GMDL(*model)
    return BasicClassifierCall(classifier, model, train, test)
