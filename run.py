from os import listdir
from os.path import join, realpath
import argparse
from multiprocessing.dummy import Pool
from itertools import repeat, product
import numpy as np
from aux.GetFold import get_fold
import pandas as pd

parser = argparse.ArgumentParser(description='Runs k-fold CV and grid search')

parser.add_argument('--path', dest='path', type=str, required=True,
                    help='the path to the datasets folds')
parser.add_argument('--pattern', dest='pattern', type=str, default='',
                    help='the pattern of files to look at')
parser.add_argument('--sets', dest='sets', action='store_true', 
                    help='only prints the sets')
parser.add_argument('--k', dest='k', type=int, default=5,
                    help='the number of folds')
parser.add_argument('--measure', dest='measure', type=int, default=0,
                    help='the index of the returned score measure to be used to decide the winner model')
parser.add_argument('--exclude', dest='exclude', nargs='+', default=[],
                    help='the list of methods that should not be ran')

args = parser.parse_args()

sets = filter(lambda x: x.find(args.pattern) >= 0, listdir(args.path))

classifiers_files = filter(lambda x: x.find('.pyc') == -1 and x.find('__init__') == -1, listdir('./classifiers/'))

classifiers = map(
  lambda x: 'classifiers.' + x.split('.')[0], 
  classifiers_files
)

classifiers = filter(
  lambda c: reduce(
    lambda x, y: x and y[0].find(y[1]) == -1,
    zip(repeat(c), args.exclude), True
  ),
  classifiers
)

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

  pool = Pool(len(models))
  data = product(models, [instance], [training], [validation])
  results = pool.map(classify, data)

  pool.close()
  pool.join()

  return results

for s in sets:
  print '{}:'.format(s)

  for module in modules:
    current_set = get_fold(args.path, s, args.k)

    instance = module.Classifier()
    print ' {}:'.format(instance)

    models = instance.models()
    test_results = []

    for _ in xrange(args.k):
      models_scores = {}
      models_descriptions = {}

      search_data = []

      for _ in xrange(args.k):
        search_data.append(next(current_set))

      pool = Pool(args.k)
      data = product(search_data, [models])
      results = pool.map(grid_search, data)

      pool.close()
      pool.join()

      results = reduce(lambda x, y: x + y, results)

      for result in results:
        model, scores = result
        key = str(model)

        if models_descriptions.has_key(key):
          models_scores[key] += np.array(scores)
        else:
          models_scores[key] = np.array(scores)
          models_descriptions[key] = model

      best = None

      for ms in models_scores:
        models_scores[ms] = models_scores[ms] / float(args.k)

        if best is None or models_scores[ms][args.measure] > models_scores[best][args.measure]:
          best = ms

      training, validation, test = search_data[-1]

      X_training, y_training = training
      X_validation, y_validation = validation

      X_training = pd.concat([X_training, X_validation])
      y_training = pd.concat([y_training, y_validation])
      training = (X_training, y_training)

      score = instance.run(models_descriptions[best], training, test)
      test_results.append(score)

    macro, micro = np.mean(test_results, axis=0)

    print '  macro-f: {}'.format(macro)
    print '  micro-f: {}'.format(micro)
