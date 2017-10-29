"""
SVM
"""

from sklearn.linear_model import SGDClassifier

class Classifier(SGDClassifier):
  def __init__(self):
    super(Classifier, self).__init__(random_state=123456789)

  def __str__(self):
    return 'SVM'

  def partial_fit(self, X, y, y_predicted, classes):
    super(Classifier, self).partial_fit(X, y, classes=classes)
