import subprocess
import os
from sklearn import preprocessing
import pandas as pd
import re

class GMDL(object):
  def __init__(self, sigma, omega):
    self.sigma = sigma
    self.omega = omega

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
      'omega': self.omega,
      'labels': labels,
      'PATH': os.environ['GMDL_PATH']
    }

    return """\
    %(PATH)s/gmdl.app \
    --stdin \
    --quiet \
    --learning_rate "0" \
    --momentum "0" \
    --tau "0" \
    --beta "0" \
    --omega "%(omega)s" \
    --forgetting_factor "1.0" \
    --sigma "%(sigma)s" \
    --label "-1" \
    --labels "%(labels)s"
    """ % variables
