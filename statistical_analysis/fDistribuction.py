# coding: utf-8
import numpy as np

critical_values = np.loadtxt('./statistical_analysis/table_fDistribuction_alpha05_2.csv',delimiter=',')

def return_critical_value(pvalue, ff_graus_liberdade_algoritmos, ff_graus_liberdade_datasets):
    """
    Returns the critical value for the two-tailed Bonferroni-Dunn test test for a given
    p-value and number of models being compared.
    """
    if pvalue == 0.05:
        col_idx = 0
    else:
        raise ValueError('p-value must be one of 0.05')
		
    return critical_values[ff_graus_liberdade_datasets-1][ff_graus_liberdade_algoritmos-1]
