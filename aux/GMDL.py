import subprocess
import os
from sklearn import preprocessing
import pandas as pd
import re

class GMDL(object):
  def __init__(self, sigma, tau):
    self.sigma = sigma
    self.tau = tau

  def fit(self, X, y):
    x = X.copy()
    x['class'] = y.values

    labels = re.sub(' +', ',', str(y.unique())[1:-1]).replace("'", '')

    self.instance = subprocess.Popen(
      self.__command(labels),
      stdout=subprocess.PIPE,
      stdin=subprocess.PIPE,
      shell=True,
      close_fds=True
    )

    data = str(x.shape[0]) + '\n' + x.to_csv(index=None, header=None)
    self.instance.stdin.write(data)

  def predict(self, X):
    x = X.copy()
    x['class'] = ''

    output, err = self.instance.communicate(x.to_csv(index=None, header=None))

    if err:
      raise Exception(err)

    return output[:-1].split(', ')

  def __command(self, labels):
    variables = {
      'sigma': self.sigma,
      'tau': self.tau,
      'labels': labels,
      'PATH': os.environ['GMDL_PATH']
    }

    return """\
    %(PATH)s/gmdl.app \
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
    """ % variables
