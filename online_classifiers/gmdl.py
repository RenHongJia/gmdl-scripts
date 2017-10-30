"""
GMDL

requires GMDL_PATH as an ENV variable
"""

from aux.GMDL import GMDL

class Classifier(GMDL):
  def __init__(self, labels=[]):
    super(Classifier, self).__init__(sigma=2, tau=0, labels=labels)

  def __str__(self):
    return 'GMDL'

  def partial_fit(self, X, y, y_predicted, classes):
    pass
