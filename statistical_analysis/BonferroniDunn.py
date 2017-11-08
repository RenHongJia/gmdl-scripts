# coding: utf-8
"""
Implements the formula to compare models with the Bonferroni-Dunn test. "The performance
of two classifiers is significantly different if the corresponding average ranks
differ by at least the critical difference" from:

  Demsar, J. "Statistical comparisons of classifiers over multiple data sets."
      The Journal of Machine Learning Research 7 (2006): 1-30.
"""
import numpy as np

#tabela baseada no livro do Zar (tab. 16) e no artigo do demsar
critical_values = np.array([
# p  0.01, 0.05, 0.10  Models
[2.576,1.960,1.645], #2
[2.807,2.242,1.960], #3
[2.936,2.394,2.128], #4
[3.024,2.498,2.242], #5
[3.091,2.576,2.327], #6
[3.144,2.639,2.394], #7
[3.189,2.690,2.450], #8 
[3.227,2.735,2.498], #9
[3.261,2.773,2.540],  #10
[3.291,2.807,2.576],  #11
[3.317,2.838,2.609],  #12
[3.342,2.866,2.639],  #13
[3.364,2.891,2.666],  #14
[3.384,2.914,2.690]  #15
])

def return_critical_value(pvalue, models):
    """
    Returns the critical value for the two-tailed Bonferroni-Dunn test test for a given
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
    Returns the critical difference for the two-tailed Bonferroni-Dunn test test for a
    given p-value, number of models being compared, and number of datasets over
    which model ranks are averaged.
    """
    cv = return_critical_value(pvalue, nModels)
    print('Critical value Bonferroni-Dunn test: %1.3f' %cv)
    cd = cv*np.sqrt( (nModels*(nModels+1))/(6*nDatasets) )
    return cd
