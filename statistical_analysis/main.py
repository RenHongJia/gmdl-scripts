# coding: utf-8
import numpy as np
import pandas as pd
import re #regular expression
import os

import sys
#sys.path.append('../')
import teste_Friedman
import nemenyi
import BonferroniDunn

import Orange
import matplotlib.pyplot as plt
import sys
import StringIO

document = ''

while True:
  line = sys.stdin.readline()

  if line == '':
    break

  document += line

document = StringIO.StringIO(document)

def import_csv_results(path,nomecolunaAvaliada):

    selectColumns = []
                
    if len(selectColumns)==0:
        fileContent = pd.read_csv( path ,sep=',')
    else:
        fileContent = pd.read_csv( path, usecols=selectColumns, sep=',')
    
    fileContent['base_dados'] = fileContent['base_dados'].str.replace(' ','') #replace em todas as linhas da coluna informada
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('_','-');
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('.mat','');
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('.wc','');
    fileContent['base_dados'] = fileContent['base_dados'].str.lower(); #convert para minusculo
     
    metodos  = list(set(fileContent['metodo']))
    
    fileContent = fileContent.sort_values(['metodo'], ascending=True)
    
    grp = fileContent.groupby(['base_dados']) #agrupa por base de dados
    #para cada grupo de bases de dados
    
    newDf = pd.DataFrame(columns=(['base_dados']+metodos))
    for nameGroup, group in grp:
        medidas = []
        for metodo in metodos:
            aux = group.loc[ group['metodo'] == metodo, nomecolunaAvaliada ].values
            if len(aux)==0 or len(aux)>1:
                print('\n\nErro!!! Dataset: %s, metodo: %s\n\n' %(nameGroup,metodo))
            else:
                medidas.append(aux[0])
                
        newDf.loc[len(newDf)+1] = [nameGroup]+medidas
    
    metodos.sort()
    selectColumns = ['base_dados'] + metodos
    newDf = newDf[selectColumns] 
    
    return newDf
    
def rank_plotting(df_Results, critical_difference, posthoc_method = 'nemenyi', control_method = 'None'):       
    res = df_Results.iloc[:,1:].values
    rank = teste_Friedman.return_rank(res)
    
    rank_mean = np.mean(rank,0) #media do rank de cada algoritmo
    idSort = np.argsort(rank_mean) #[::-1] reverte a ordem dos indices para ordenar por ordem decrescente
    rank = rank[:,idSort]
    
    metodos = list(df_Results.columns[1:])
    metodos = [metodos[i] for i in idSort]
    
    if posthoc_method == 'nemenyi':
        teste_Friedman.plota_CD(rank, metodos, critical_difference, pathFile='fig_Nemenyi_1.eps')
        
        Orange.evaluation.graph_ranks(rank_mean, metodos, cd=critical_difference, width=8, textspace=2.0)
        plt.savefig('fig_Nemenyi_2.eps', format='eps', bbox_inches='tight') #bbox_inches='tight' to save legend outside the plot

    elif posthoc_method == 'dunn':
        teste_Friedman.plota_CD2(rank, metodos, critical_difference, metodoInteresse = control_method, pathFile='fig_BonferroniDunn.eps', formatFile='eps')

if __name__ == "__main__":
    
    path = 'example_results.csv'
    nomecolunaAvaliada = 'F-medidaMacro' 
    
    alpha = 0.05
    
    df_Results = import_csv_results(document,nomecolunaAvaliada)
    nDatasets = df_Results.shape[0]
    nModels = df_Results.shape[1]-1    

    print('\n\n***Friedman Test***');    
    teste_Friedman.perform_Friedman_test(df_Results, alpha)
    
    print('\n\n***Nemesis Test***');
    CD_nemenyi = nemenyi.critical_difference(alpha, nModels, nDatasets)  
    
    print('\n***Bonferroni-Dunn test***');
    CD_Dunn = BonferroniDunn.critical_difference(alpha, nModels, nDatasets)
 
    print('\n***Bonferroni-Dunn plot***');
    rank_plotting(df_Results, CD_Dunn, posthoc_method = 'dunn', control_method = 'GMDL')
   
    print('\n***Nemenyi plot***');     
    rank_plotting(df_Results, CD_nemenyi, posthoc_method = 'nemenyi')
    
    
    

  
        
