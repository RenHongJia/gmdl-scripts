import numpy as np

def macro(cf):
  tp, fn, fp = get_ratios(cf)
  C = len(tp)

  if C == 0:
    return np.nan, np.nan, np.nan

  precision = map(lambda i: div(tp[i], tp[i] + fp[i]), xrange(C))
  precision = sum(precision) / C

  recall = map(lambda i: div(tp[i], tp[i] + fn[i]), xrange(C))
  recall = sum(recall) / C

  precision = precision if not np.isnan(precision) else 0
  recall = recall if not np.isnan(recall) else 0

  return precision, recall, harmonic_mean(precision, recall)

def micro(cf):
  tp, fn, fp = get_ratios(cf)
  C = len(tp)

  if C == 0:
    return np.nan, np.nan, np.nan

  precision = div(sum(tp), sum(map(lambda i: tp[i] + fp[i], xrange(C))))
  recall = div(sum(tp), sum(map(lambda i: tp[i] + fn[i], xrange(C))))

  precision = precision if not np.isnan(precision) else 0
  recall = recall if not np.isnan(recall) else 0

  return precision, recall, harmonic_mean(precision, recall)

def get_ratios(cf):
  matrix = np.array(cf.as_matrix())

  tp = matrix.diagonal()
  fn = matrix.sum(axis=1) - tp
  fp = matrix.sum(axis=0) - tp

  return tp, fn, fp

def harmonic_mean(a, b):
  return 2 * ((a * b) / (a + b))

def div(a, b):
  return 0 if b == 0 else (float(a) / float(b))
