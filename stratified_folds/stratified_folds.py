import argparse
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold, train_test_split

parser = argparse.ArgumentParser(description='Creates K-folds files for training/testing')

parser.add_argument('--path', dest='path', type=str, required=True,
                    help='the path to the datasets')
parser.add_argument('--output', dest='output', type=str, required=True,
                    help='the path where to dump')
parser.add_argument('--pattern', dest='pattern', type=str, default='',
                    help='the pattern of files to look at')
parser.add_argument('--sets', dest='sets', action='store_true', 
                    help='only prints the sets')
parser.add_argument('--k', dest='k', type=int, default=5,
                    help='the number of folds')
parser.add_argument('--label', dest='label', type=int, default=[], nargs='*',
                    help='the column representing the label')

args = parser.parse_args()

sets = [f for f in listdir(args.path) if isfile(join(args.path, f))]
sets = filter(lambda x: x.find(args.pattern) >= 0, sets)

if args.sets:
  print sets
  exit(0)

labels = args.label + ([None] * (len(sets) - len(args.label)))

for i in xrange(0, len(sets)):
  s = sets[i]
  X = pd.read_csv(join(args.path, s), header=None)

  label = (len(X.columns) - 1) if labels[i] is None else labels[i]

  Y = X[label]
  del X[label]

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12345, stratify=Y)

  X_train = X_train.reset_index(drop=True)
  y_train = y_train.reset_index(drop=True)

  skf = StratifiedKFold(n_splits=args.k)
  skf.get_n_splits(X_train, y_train)

  k = 0

  set_path = join(args.output, s)
  os.makedirs(set_path)

  X_test.assign(label=y_test.values).to_csv(join(set_path, 'test'), header=False)
  X_train.assign(label=y_train.values).to_csv(join(set_path, 'train'), header=False)

  for train_index, validation_index in skf.split(X_train, y_train):
    k += 1

    foldX_train, foldX_validation = X_train.loc[train_index], X_train.loc[validation_index]
    foldY_train, Y_validation = y_train.loc[train_index], y_train.loc[validation_index]
    
    path = join(set_path, str(k))

    os.makedirs(path)

    foldX_train.assign(label=foldY_train.values).to_csv(join(path, 'train'), header=False)
    foldX_validation.assign(label=Y_validation.values).to_csv(join(path, 'validation'), header=False)

