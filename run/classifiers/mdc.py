"""
MDC

requires MDC_PATH as an ENV variable
"""

import subprocess
from itertools import product
import os
from sklearn import preprocessing
import pandas as pd

class Classifier(object):
  def __init__(self):
    print 'MDC'
  
  def models(self):
    sigma = [0.5, 1, 2, 3]
    omega = [8, 16, 32, 64]

    return list(product(sigma, omega))

  def normalize(self, id, path, train, test):
    X_train = pd.read_csv(os.path.join(path, train), header=None)
    X_test = pd.read_csv(os.path.join(path, test), header=None)
    label = len(X_train.columns) - 1

    y_train = X_train[label]
    del X_train[label]

    y_test = X_test[label]
    del X_test[label]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    path = '/tmp/'
    train = 'train.mdc' + str(id)
    test = 'test.mdc' + str(id)

    X_train.assign(label=y_train.values).to_csv(path + train, header=False)
    X_test.assign(label=y_test.values).to_csv(path + test, header=False)

    return path, train, test

  def run(self, model, path, train, test, set_name, id=0):
    path, train, test = self.normalize(id, path, train, test)

    cmd = self.__command(locals())

    p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, shell=True
    )

    output, err = p.communicate()

    try:
      scores = map(lambda x: float(x), output.split('/'))
    except:
      raise Exception(cmd)

    return scores

  def __command(self, variables):
    variables['sigma'] = variables['model'][0]
    variables['omega'] = variables['model'][1] 
    variables['PATH'] = os.environ['MDC_PATH']
    variables['CONFIG'] = os.path.realpath('./metadata/mdc.config.json')

    return """\
    %(PATH)s/mdc.app \
    --fscore \
    --quiet \
    --no-incremental-learning \
    --training "%(train)s" \
    --testing "%(test)s" \
    --learning_rate "0" \
    --momentum "0" \
    --tau "0" \
    --beta "0" \
    --omega "%(omega)s" \
    --forgetting_factor "1.0" \
    --sigma "%(sigma)s" \
    --label "-1" \
    --set "%(set_name)s" \
    --path "%(path)s" \
    --config "%(CONFIG)s"
    """ % variables
    
