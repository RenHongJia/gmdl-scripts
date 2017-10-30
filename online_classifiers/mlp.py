"""
Multi-Layer Perceptron
"""

from sklearn.neural_network import MLPClassifier
import warnings

class Classifier(MLPClassifier):
  def __init__(self, labels=[]):
    super(Classifier, self).__init__(random_state=123456789)

  def __str__(self):
    return 'MLP'

  def partial_fit(self, X, y, y_predicted, classes):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      super(Classifier, self).partial_fit(X, y, classes=classes)
