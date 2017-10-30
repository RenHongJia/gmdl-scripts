import subprocess
import os
from sklearn import preprocessing
import pandas as pd
import re
import time

TRAINING_TOKEN = '<Training>\n'
CORRECTION_TOKEN = '<Correction>\n'
TEST_TOKEN = '<Test>\n'

class GMDL(object):
  def __init__(self, sigma, tau, online=False, labels=[]):
    self.sigma = sigma
    self.tau = tau
    self.labels = labels
    self.online = online

  def __del__(self):
    if self.instance is not None:
      self.instance.kill()

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
      csv = x.to_csv(index=None, header=None)
      csv = csv.replace('\n', '\n' + TRAINING_TOKEN)
      data = TRAINING_TOKEN + csv[:-len(TRAINING_TOKEN)]
    else:
      data = str(n) + '\n' + x.to_csv(index=None, header=None)

    self.instance.stdin.write(data)

  def partial_fit(self, X, y, y_predicted):
    x = X.copy()
    x['class'] = y[0]

    PREDICTION_TOKEN = str(y_predicted) + '\n'

    csv = x.to_csv(index=None, header=None)
    data = CORRECTION_TOKEN + csv + PREDICTION_TOKEN

    self.instance.stdin.write(data)

  def predict(self, X):
    x = X.copy()
    x['class'] = ''

    if self.online:
      csv = x.to_csv(index=None, header=None)
      data = TEST_TOKEN + csv
      
      self.instance.stdin.write(data)

      time.sleep(0.01)

      output = self.instance.stdout.readline().rstrip()

      return [output]
    
    output, err = self.instance.communicate(x.to_csv(index=None, header=None))
    self.instance = None

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
