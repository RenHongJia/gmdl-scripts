from os import listdir
from os.path import join, realpath
import argparse
from multiprocessing.dummy import Pool
from itertools import repeat, product
import numpy as np
from aux.GetFold import get_fold
from aux.Metrics import macro, micro
import pandas as pd

parser = argparse.ArgumentParser(description='Runs k-fold CV and grid search')

parser.add_argument('--path', dest='path', type=str, required=True,
                    help='the path to the datasets folds')
parser.add_argument('--exclude-sets', dest='exclude_sets', nargs='+', default=[], help='the list of sets that should not be ran')
parser.add_argument('--only-sets', dest='only_sets', nargs='+', default=[], help='the list of sets that should be ran (prioritized)')
parser.add_argument('--sets', dest='sets', action='store_true', 
                    help='only prints the sets')
parser.add_argument('--k', dest='k', type=int, default=5,
                    help='the number of folds')
parser.add_argument('--measure', dest='measure', type=str, default='macro-f',
                    help='the index of the returned score measure to be used to decide the winner model')
parser.add_argument('--exclude', dest='exclude', nargs='+', default=[],
                    help='the list of methods that should not be ran')
parser.add_argument('--only', dest='only', nargs='+', default=[],
                    help='the list of methods that should be ran only (prioritized)')

args = parser.parse_args()

sets = set(listdir(args.path)) - set(args.exclude_sets)
sets = args.only_sets if len(args.only_sets) > 0 else sets
sets = list(sets)

classifiers_files = filter(lambda x: \
  x.find('.pyc') == -1 and x.find('__init__') == -1, \
  listdir('./classifiers/') \
)

classifiers_files = map(lambda x: x.split('.')[0], classifiers_files)
classifiers_files = set(classifiers_files) - set(args.exclude)
classifiers_files = args.only if len(args.only) > 0 else classifiers_files
classifiers = map(lambda x: 'classifiers.' + x, list(classifiers_files))

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

def classify(data):
  model, instance, training, validation = data
  return (model, instance.run(model, training, validation))

def grid_search(data):
  search_data, models = data
  training, validation, test = search_data

  pool = Pool(args.k)
  data = product(models, [instance], [training], [validation])
  results = pool.map(classify, data)

  pool.close()
  pool.join()

  return results

def compute_outer_fold(data):
  search_data, models = data
  confusion_matrices = {}
  models_results = {}
  classes = None

  pool = Pool(args.k)
  data = product(search_data, [models])
  results = pool.map(grid_search, data)

  pool.close()
  pool.join()

  results = reduce(lambda x, y: x + y, results)

  for result in results:
    model, confusion_matrix = result

    if classes is None:
      classes = confusion_matrix.columns.tolist()

    confusion_matrix = confusion_matrix.reindex_axis(classes, axis=1)

    key = str(model)

    if confusion_matrices.has_key(key):
      confusion_matrices[key] += confusion_matrix
    else:
      confusion_matrices[key] = confusion_matrix
      models_results[key] = {'model': model}

  for model in confusion_matrices:
    p_macro, r_macro, f_macro = macro(confusion_matrices[key])
    p_micro, r_micro, f_micro = micro(confusion_matrices[key])

    models_results[model]['macro-f'] = f_macro
    models_results[model]['micro-f'] = f_micro

  best = None

  for model in confusion_matrices:
    if best is None:
      best = model
    else:
      best_score = models_results[best][args.measure]
      current_score = models_results[model][args.measure]

      if current_score > best_score:
        best = model

  training, validation, test = search_data[-1]

  X_training, y_training = training
  X_validation, y_validation = validation

  X_training = pd.concat([X_training, X_validation])
  y_training = pd.concat([y_training, y_validation])
  training = (X_training, y_training)

  return instance.run(models_results[best]['model'], training, test)

for s in sets:
  print '{}:'.format(s)

  for module in modules:
    current_set = get_fold(args.path, s, args.k)

    instance = module.Classifier()
    print ' {}:'.format(instance)

    models = instance.models()

    search_data = []

    for _ in xrange(args.k ** 2):
      search_data.append(next(current_set))

    search_data = [ \
      search_data[(i * args.k) : (i * args.k + args.k)] \
      for i in xrange(args.k) \
    ]

    pool = Pool(args.k)
    data = product(search_data, [models])
    results = pool.map(compute_outer_fold, data)

    pool.close()
    pool.join()

    final_cm = None
    cm_counter = 1

    for cm in results:
      print '  fold-{}: |'.format(str(cm_counter))
      print '    ' + cm.to_string().replace('\n', '\n    ')

      cm_counter += 1

      if final_cm is None:
        final_cm = cm
      else:
        final_cm += cm.reindex_axis(final_cm.columns.tolist(), axis=1)

    p_macro, r_macro, f_macro = macro(final_cm)
    p_micro, r_micro, f_micro = micro(final_cm)

    print '  macro:'
    print '    - precision: {}'.format(p_macro)
    print '    - recall: {}'.format(r_macro)
    print '    - f1: {}'.format(f_macro)
    print '  micro:'
    print '    - precision: {}'.format(p_micro)
    print '    - recall: {}'.format(r_micro)
    print '    - f1: {}'.format(f_micro)
