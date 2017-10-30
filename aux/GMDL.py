import subprocess
import os
from sklearn import preprocessing
import pandas as pd
import re

class GMDL(object):
  def __init__(self, sigma, tau, online=False, labels=[]):
    self.sigma = sigma
    self.tau = tau
    self.labels = labels
    self.online = online

  def fit(self, X, y):
    x = X.copy()
    x['class'] = y.values

    n, m = X.shape

    label_set = self.labels if len(self.labels) > 0 else y.unique()
    labels = re.sub(' +', ',', str(label_set)[1:-1]).replace("'", '')

    self.instance = subprocess.Popen(
      self.__command(labels, m),
      stdout=subprocess.PIPE,
      stdin=subprocess.PIPE,
      shell=True,
      close_fds=True
    )

    if self.online:
      pass
    else:
      data = str(n) + '\n' + x.to_csv(index=None, header=None)

    self.instance.stdin.write(data)

  def partial_fit(X, y, y_predicted):
    pass

  def predict(self, X):
    x = X.copy()
    x['class'] = ''

    output, err = self.instance.communicate(x.to_csv(index=None, header=None))

    if err:
      raise Exception(err)

    return output[:-1].split(', ')

  def __command(self, labels, dimension):
    variables = {
      'online': '--online' if self.online else '',
      'sigma': self.sigma,
      'tau': self.tau,
      'labels': labels,
      'dimension': dimension,
      'PATH': os.environ['GMDL_PATH']
    }

    return """\
    %(PATH)s/gmdl.app \
    %(online)s \
    --stdin \
    --quiet \
    --learning_rate "0.001" \
    --momentum "0.9" \
    --tau "%(tau)s" \
    --sigma "%(sigma)s" \
    --labels "%(labels)s" \
    --omega "32" \
    --forgetting_factor "1.0" \
    --label "-1" \
    --dimension "%(dimension)s" \
    """ % variables
