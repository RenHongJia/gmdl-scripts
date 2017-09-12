from os.path import join
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_fold(path, current_set, k):
  X = pd.read_csv(join(path, current_set), header=None)
  label = len(X.columns) - 1

  Y = X[label]
  del X[label]

  outer_skf = StratifiedKFold(n_splits=k)
  outer_skf.get_n_splits(X, Y)

  outer_folds = outer_skf.split(X, Y)

  for training_idx, test_idx in outer_folds:
    X_outer_train, X_test = X.loc[training_idx], X.loc[test_idx]
    y_outer_train, y_test = Y.loc[training_idx], Y.loc[test_idx]

    X_outer_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    y_outer_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    inner_skf = StratifiedKFold(n_splits=k)
    inner_skf.get_n_splits(X_outer_train, y_outer_train)

    inner_folds = inner_skf.split(X_outer_train, y_outer_train)

    for inner_training_idx, validation_idx in inner_folds:
      X_train = X_outer_train.loc[inner_training_idx]
      X_validation = X_outer_train.loc[validation_idx]

      y_train = y_outer_train.loc[inner_training_idx]
      y_validation = y_outer_train.loc[validation_idx]

      yield (X_train, y_train), (X_validation, y_validation), (X_test, y_test)
