# coding: utf-8
import numpy as np
import pandas as pd

critical_values = pd.read_csv('./statistical_analysis/table_Zar-2009-friedmanX2.csv', sep=',')

def return_critical_value(pvalue, nModels, nDatasets):
    """
    Returns the critical value for the two-tailed Bonferroni-Dunn test test for a given
    p-value and number of models being compared.
    """
    if pvalue == 0.05:
        col_idx = 3
    else:
        raise ValueError('p-value must be one of 0.05')

    auxCriticalValue = 0;
    for i in range(len(critical_values)):
        if (critical_values['number_Models'][i] == nModels) and (critical_values['number_Datasets'][i] == nDatasets):
            auxCriticalValue = critical_values['criticalValue'][i];        

    if auxCriticalValue!=0:	
        return auxCriticalValue
    else:
        raise ValueError('\n\nError: Não ha registro do valor critico para essa quantidade de modelos e datasets')







##critical_values = np.genfromtxt('table_Zar-2009-friedmanX2.csv',delimiter=',',names=True)
##
##def return_critical_value(pvalue, nModels, nDatasets):
##    """
##    Returns the critical value for the two-tailed Bonferroni-Dunn test test for a given
##    p-value and number of models being compared.
##    """
##    if pvalue == 0.05:
##        col_idx = 3
##    else:
##        raise ValueError('p-value must be one of 0.05')
##
##    auxCriticalValue = 0;
##    for i in range(len(critical_values)):
##        if (critical_values['number_Models'][i] == nModels) and (critical_values['number_Datasets'][i] == nDatasets):
##            auxCriticalValue = critical_values['criticalValue'][i];        
##
##    if auxCriticalValue!=0:	
##        return auxCriticalValue
##    else:
##        raise ValueError('\n\nError: Não há registro do valor crítico para essa quantidade de modelos e datasets')
