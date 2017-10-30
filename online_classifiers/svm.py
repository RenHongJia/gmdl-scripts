"""
SVM
"""

from sklearn.linear_model import SGDClassifier
import warnings

class Classifier(SGDClassifier):
  def __init__(self, labels=[]):
    super(Classifier, self).__init__(random_state=123456789)

  def __str__(self):
    return 'SVM'

  def partial_fit(self, X, y, y_predicted, classes):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      super(Classifier, self).partial_fit(X, y, classes=classes)
