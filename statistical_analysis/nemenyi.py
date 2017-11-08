# coding: utf-8
"""
Implements the formula to compare models with the Nemenyi test. "The performance
of two classifiers is significantly different if the corresponding average ranks
differ by at least the critical difference" from:

  Demsar, J. "Statistical comparisons of classifiers over multiple data sets."
      The Journal of Machine Learning Research 7 (2006): 1-30.
"""
import numpy as np

critical_values = np.array([
# p  0.01, 0.05   0.10  Models
[2.576,1.960,1.645], #2
[2.913,2.343,2.052], #3
[3.113,2.569,2.291], #4
[3.255,2.728,2.459], #5
[3.364,2.850,2.589], #6 
[3.452,2.949,2.693], #7
[3.526,3.031,2.780], #8
[3.590,3.102,2.855], #9
[3.646,3.164,2.920], #10
[3.696,3.219,2.978], #11
[3.741,3.268,2.030], #12
[3.781,3.313,3.077], #13
[3.818,3.354,3.120], #14
[3.853,3.391,3.159], #15
[3.884,3.426,3.196], #16
[3.914,3.458,3.230], #17
[3.941,3.489,3.261], #18
[3.967,3.517,3.291], #19
[3.992,3.544,3.319] #20
])

def return_critical_value(pvalue, models):
    """
    Returns the critical value for the two-tailed Nemenyi test for a given
    p-value and number of models being compared.
    """
    if pvalue == 0.01:
        col_idx = 0
    elif pvalue == 0.05:
        col_idx = 1
    elif pvalue == 0.10:
        col_idx = 2
    else:
        raise ValueError('p-value must be one of 0.05, or 0.10')

    if not (2 <= models and models <= 50):
        raise ValueError('number of models must be in range [2, 10]')
    else:
        row_idx = models - 2

    return critical_values[row_idx][col_idx]

def critical_difference(pvalue, nModels, nDatasets):
    """
    Returns the critical difference for the two-tailed Nemenyi test for a
    given p-value, number of models being compared, and number of datasets over
    which model ranks are averaged.
    """
    cv = return_critical_value(pvalue, nModels)
    print('Critical value Nemenyi test: %1.3f' %cv)
    cd = cv*np.sqrt( (nModels*(nModels+1))/(6*nDatasets) )
    return cd
