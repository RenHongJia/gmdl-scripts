from os import listdir
from os.path import join, realpath
import argparse
from multiprocessing.dummy import Pool
from itertools import repeat, product
import numpy as np
from aux.GetFold import get_fold
from aux.Metrics import macro, micro
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='Runs k-fold CV and grid search')

parser.add_argument('--path', dest='path', type=str, required=True,
                    help='the path to the datasets folds')
parser.add_argument('--exclude-sets', dest='exclude_sets', nargs='+', default=[], help='the list of sets that should not be ran')
parser.add_argument('--only-sets', dest='only_sets', nargs='+', default=[], help='the list of sets that should be ran (prioritized)')
parser.add_argument('--sets', dest='sets', action='store_true', 
                    help='only prints the sets')
parser.add_argument('--exclude', dest='exclude', nargs='+', default=[],
                    help='the list of methods that should not be ran')
parser.add_argument('--only', dest='only', nargs='+', default=[],
                    help='the list of methods that should be ran only (prioritized)')
parser.add_argument('--pool', dest='pool', type=int, default=5,
                    help='the number of threads to have')
parser.add_argument('--training_size', dest='training_size', type=str, default='10%', help='the number of samples to train: either a number or percentage')
parser.add_argument('--normalize', dest='normalize', action='store_true', 
                    help='whether or not it should normalize the data')
parser.add_argument('--feedback', dest='feedback', type=str, required=False, default='never',
                    help='how long until give feedback: "never", "[0.0-1.0]" for random, "int" for number of epochs or "^int" for exponential epochs')

args = parser.parse_args()

sets = set(listdir(args.path)) - set(args.exclude_sets)
sets = args.only_sets if len(args.only_sets) > 0 else sets
sets = list(sets)

classifiers_files = filter(lambda x: \
  x.find('.pyc') == -1 and x.find('__init__') == -1, \
  listdir('./online_classifiers/') \
)

classifiers_files = map(lambda x: x.split('.')[0], classifiers_files)
classifiers_files = set(classifiers_files) - set(args.exclude)
classifiers_files = args.only if len(args.only) > 0 else classifiers_files
classifiers = map(lambda x: 'online_classifiers.' + x, list(classifiers_files))

def set_importance(name):
  df = pd.read_csv(join(args.path, name))
  return df.shape[0] * df.shape[1]

sets = sorted(sets, key=set_importance)

if args.sets:
  print sets
  exit(0)

modules = map(
  lambda x: __import__(x, fromlist=['Classifier']),
  classifiers
)

test_size = 0
FEEDBACK = {
  'never': args.feedback == 'never',
  'proba': args.feedback.find('.') != -1,
  'expon': args.feedback[0] == '^'
}

if args.training_size.find('%') > 0:
  test_size = 1 - (float(args.training_size[:-1]) / 100)
else:
  test_size = int(args.training_size)

def should_give_feedback(epoch, head_epoch, n_mistakes):
  if FEEDBACK['never']:
    return False

  if FEEDBACK['expon']:
    return (epoch - head_epoch) >= np.floor(float(args.feedback[1:]) ** (n_mistakes))

  if FEEDBACK['proba']:
    return np.random.random() <= float(args.feedback)

  return (epoch - head_epoch) == int(args.feedback)

def normalize(X):
  X_aux = []
  scaler = preprocessing.StandardScaler().fit(X)

  for i in xrange(X.shape[0]):
    sample = X.iloc[i].values.reshape(1, -1)
    normalized_sample = scaler.transform(sample)
    X_aux.append(normalized_sample[0])
    scaler = scaler.partial_fit(sample)

  return pd.DataFrame(X_aux)

def predict(data):
  module, sets = data
  X_train, X_test, y_train, y_test, labels = sets

  classifier = module.Classifier(labels=labels)
  classifier.partial_fit(X_train, y_train, y_train, labels)

  y_pred = []
  y_mistake = []
  epoch = 0
  n_mistakes = 0

  for i in xrange(X_test.shape[0]):
    epoch += 1

    sample = pd.DataFrame(X_test.iloc[i].values.reshape(1, -1))
    yi_true = y_test.iloc[i]
    yi_pred = classifier.predict(sample)
    yi_pred = yi_pred[0]

    y_pred.append(yi_pred)

    if yi_pred != yi_true:
      n_mistakes += 1
      y_mistake.append((epoch, sample, yi_true, yi_pred))

    if len(y_mistake) > 0 and should_give_feedback(epoch, y_mistake[0][0], n_mistakes):
      _, _sample, _yi_true, _yi_pred = y_mistake.pop(0)
      classifier.partial_fit(_sample, [_yi_true], [_yi_pred], labels)

  y_pred = pd.Series(y_pred, dtype=y_test.dtype)
  cm = confusion_matrix(y_test, y_pred, labels=labels)

  return (str(classifier), pd.DataFrame(cm, columns=labels))

for s in sets:
  print '{}:'.format(s)

  X = pd.read_csv(join(args.path, s), header=None)
  label = len(X.columns) - 1

  y = X[label]
  del X[label]

  labels = np.unique(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123456789)

  if args.normalize:
    X_train = normalize(X_train)
    X_test = normalize(X_test)

  pool = Pool(len(modules))
  data = product(modules, [(X_train, X_test, y_train, y_test, labels)])
  results = pool.map(predict, data)

  pool.close()
  pool.join()

  results = sorted(results, key=lambda x: x[0])

  for result in results:
    name, cm = result

    print '  {}:'.format(name)
    print '    confusion_matrix: |'
    print '      ' + cm.to_string().replace('\n', '\n      ')

    p_macro, r_macro, f_macro = macro(cm)
    p_micro, r_micro, f_micro = micro(cm)

    print '    macro:'
    print '      - precision: {}'.format(p_macro)
    print '      - recall: {}'.format(r_macro)
    print '      - f1: {}'.format(f_macro)
    print '    micro:'
    print '      - precision: {}'.format(p_micro)
    print '      - recall: {}'.format(r_micro)
    print '      - f1: {}'.format(f_micro)
