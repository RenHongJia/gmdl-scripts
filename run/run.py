from os import listdir
from os.path import join, realpath
import argparse
from multiprocessing.dummy import Pool
from itertools import repeat
import numpy as np

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

def call_classifier(data):
  instance, model, i = data

  path = realpath(join(args.path, s, str(i))) + '/'

  scores = instance.run(model, path, 'train', 'validation', s, i)

  return i, scores

def call_per_model(model):
  final_score = None

  pool = Pool(args.k)
  data = zip(repeat(instance), repeat(model), xrange(1, args.k + 1))
  results = pool.map(call_classifier, data)

  pool.close() 
  pool.join()

  final_score = None

  for r in results:
    final_score = r[1] if final_score is None else np.array(final_score) + np.array(r[1])

  final_score = [x / args.k for x in final_score]

  return [model, final_score]

def line(symbol):
  print symbol * 80

for s in sets:
  line('=')
  print 'SET', s
  line('=')

  for module in modules:
    line('-')
    instance = module.Classifier()
    line('-')

    models = instance.models()

    pool = Pool(len(models))
    results = pool.map(call_per_model, models)

    pool.close()
    pool.join()

    winner_score = 0
    winner_model = None

    for r in results:
      model, final_score = r

      print 'model', model, 'average', final_score

      if winner_model is None or final_score[args.measure] > winner_score:
        winner_score = final_score[args.measure]
        winner_model = model

    print ''
    line('*')

    path = realpath(join(args.path, s)) + '/'
    print 'winner model', winner_model, 'score', winner_score, '=>', instance.run(winner_model, path, 'train', 'test', s)

    line('*')
    print ''
