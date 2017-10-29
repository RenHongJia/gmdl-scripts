"""
GMDL

requires GMDL_PATH as an ENV variable
"""

from aux.GMDL import GMDL

class Classifier(GMDL):
  def __init__(self):
    super(Classifier, self).__init__(sigma=2, tau=0)

  def __str__(self):
    return 'GMDL'
