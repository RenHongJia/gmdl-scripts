# coding: utf-8
import numpy as np
import pandas as pd

critical_values = pd.read_csv('./statistical_analysis/table_x2Distribuction.csv', sep=',')

def return_critical_value(pvalue, nModels):
    """
    Returns the critical value for the two-tailed Bonferroni-Dunn test test for a given
    p-value and number of models being compared.
    """
    
    degrees_freedom = nModels-1;
    coluna_csv = 'p='+str(pvalue)
    
    return critical_values[coluna_csv].values[degrees_freedom-1] 
