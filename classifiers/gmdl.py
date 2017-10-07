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
    beta = [-8, -16, -32, -64]
    tau = [0, 1, 2, 5]
    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    momentum = [0.9, 0.09, 0.009]

    return list(product(sigma, omega, beta, tau, learning_rate, momentum))

  def run(self, model, train, test):
    classifier = GMDL(*model)
    return BasicClassifierCall(classifier, model, train, test)
