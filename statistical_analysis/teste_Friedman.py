# coding: utf-8
import numpy as np
import scipy as sp
import scipy.stats

import pandas as pd

import nemenyi
import BonferroniDunn
import fDistribuction
import x2Distribuction
import x2Distribuction_Tabela_Zar_2009

import matplotlib.pyplot as plt

def return_rank(res):
    rank = np.zeros( (res.shape) )
    for i in range( len(res[:,1]) ): 
        rank[i,:] = sp.stats.rankdata( res[i,:] ) #rank de cada metodo para cada dataset i
        rank[i,:] = (len(rank[i,:])+1)-rank[i,:]; #invertendo o rank para que o melhor metodo fica com valor 1, o segundo melhor com valor 2, etc
        
    return rank


def perform_Friedman_test(auxRes, alpha):

    res = auxRes.iloc[:,1:].values
    res = np.round(res,3) 
    #print(res)
    
    n = len(res[:,1]) #numero de datasetsnewDf
    k = len(res[1,:]) #numero de algoritmos
    #print('\nQtd. de datasets: ',n,' Qtd. de algoritmos: ',k)
    
    rank = return_rank(res) #retorna os ranks de cada algoritmo
    
    R = np.mean(rank,0) #media do rank de cada algoritmo
    #print('R:',R)
    
    x2 = ( (12*n)/(k*(k+1)) ) * ( sum(R**2) - (k*(k+1)**2)/4 )
    ff_graus_liberdade_algoritmos = k-1
    ff_graus_liberdade_datasets = (k-1)*(n-1)
    #print('\nGraus de liberdade:(%d,%d)' % (ff_graus_liberdade_algoritmos,ff_graus_liberdade_datasets))
    print('\nx2: %1.3f' % x2)
    
    
    #print('\n\n\n***Friedman Test***');
    critical_value_x2 = x2Distribuction.return_critical_value(alpha,k) 
    
    print('\ncritical_value_x2: %1.3f' % critical_value_x2)
    #print('critical_value_friedman [Tabela Zar]: %1.3f' % critical_value_friedman)
    
    if(x2<critical_value_x2):
        print('\nValor critico de x2 "maior" que x2. Portanto, a hipotese nula de igualdade entre os metodos "nao pode ser rejeitada".')
    else:
        print('\nValor critico de x2 "menor" que x2. Portanto, a hipotese nula de igualdade entre os metodos deve ser "rejeitada".')
    
    #if(x2<critical_value_friedman):
    #    print('\nValor critico de Friedman "maior" que x2. Portanto, a hipotese nula de igualdade entre os metodos "nao pode ser rejeitada".')
    #else:
    #    print('\nValor critico de Friedman "menor" que x2. Portanto, a hipotese nula de igualdade entre os metodos deve ser "rejeitada".')
    
    
    
    
    ###Melhoria do friedman (Iman and Davenport (1980))
    ##print('\n\n\n***Friedman Test (Iman and Davenport (1980)***');
    ##ff = ((n-1)*x2)/( n*(k-1)-x2 )
    ##critical_value_ff = fDistribuction.return_critical_value(alpha, ff_graus_liberdade_algoritmos, ff_graus_liberdade_datasets);
    ##print('\nff: %1.3f -- Critical_value_ff: %1.3f' % (ff, critical_value_ff))
    ##      
    ##if(ff<critical_value_ff):
    ##    print('\nValor critico de ff "maior" que ff. Portanto, a hipotese nula de igualdade entre os metodos "nao pode ser rejeitada".')
    ##else:
    ##    print('\nValor critico de ff "menor" que ff. Portanto, a hipotese nula de igualdade entre os metodos deve ser "rejeitada".')
        
        
        
        
def imprimi_postHoc_test(CD, R, methods, metodoInteresse):
      
    metodos_superiores=[]; metodos_inferiores=[]; metodos_equivalentes=[];  
    for i in range( 0,len(R) ):
        for j in range( i+1,len(R) ):
            if abs(R[i]-R[j]) > CD:
                if( metodoInteresse == methods[i]):
                    if (R[i]<R[j]):
                        metodos_inferiores.append( methods[j] ) 
                    else:
                        metodos_superiores.append( methods[j] ) 
                if( metodoInteresse == methods[j]):
                    if (R[i]<R[j]):
                        metodos_superiores.append( methods[i] ) 
                    else:
                        metodos_inferiores.append( methods[i] ) 
            else:
                if( metodoInteresse == methods[i]):
                    metodos_equivalentes.append(methods[j])    
                if( metodoInteresse == methods[j]):
                    metodos_equivalentes.append(methods[i])   
                
    print('\nSuperiores ao ',metodoInteresse, ': ',metodos_superiores);
    print('Equivalentes ao ',metodoInteresse,  ': ',metodos_equivalentes)
    print('Inferiores ao ',metodoInteresse,  ': ',metodos_inferiores)
    
    
def imprimi_holmProcedure(alpha, R, k, n, methods, metodoInteresse):
        
    i = methods.index(metodoInteresse)
    
    print('\n')
    metodos_superiores=[]; metodos_inferiores=[]; metodos_equivalentes=[]; 
    
    z = np.zeros(k)
    p_value = np.zeros(k)
    for j in range( 0,len(R) ):
        if j!=i: 
            z[j] = ( abs(R[i]-R[j]) ) / ( np.sqrt( (k*(k+1))/(6*n) ) )
            p_value[j] = scipy.stats.norm.sf(abs(z[j]))*2 #twosided
                
    idxSort = np.argsort(p_value) #ordena por ranking
    
    R = R[idxSort]
    z = z[idxSort]
    p_value = p_value[idxSort]
    
    sort_methods=[]
    for i in range(k):
        sort_methods.append(methods[idxSort[i]]) 
        
    methods = sort_methods
    i = methods.index(metodoInteresse)

    print(methods)    
    
    i2 = 1;     
    for j in range( 0,len(R) ):    
        if j!=i:        
            adjusted_alpha = alpha/(k-i2)
            print('z: ',round(z[j],3), ' -- p: ', round(p_value[j],3), ' -- adjusted_alpha: ',round(adjusted_alpha,3) )
            if p_value[i2] < adjusted_alpha:
                if (R[i]<R[j]):
                    metodos_inferiores.append( methods[j] ) 
                else:
                    metodos_superiores.append( methods[j] ) 
            else:
                metodos_equivalentes.append(methods[j]) 
                
            i2+=1;
                
    print('\nSuperiores ao ',metodoInteresse, ': ',metodos_superiores);
    print('Equivalentes ao ',metodoInteresse,  ': ',metodos_equivalentes)
    print('Inferiores ao ',metodoInteresse,  ': ',metodos_inferiores)
        
def plota(rank, metodos):

    x = np.arange(0,len(metodos))
    rank_mean = np.mean(rank,0) #media do rank de cada algoritmo  
    rank_std = np.std(rank,0) #desvio padrao do rank de cada algoritmo
    rank_min = np.min(rank,0) #desvio padrao do rank de cada algoritmo
    rank_max = np.max(rank,0) #desvio padrao do rank de cada algoritmo

    asymmetric_error = [rank_mean-rank_min,rank_max-rank_mean]
    
    plt.errorbar(x, rank_mean, yerr=asymmetric_error, linestyle='None', marker='o', color='black')
    plt.xticks(x, metodos)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, ha='right') #rotate x labels #rotate x labels #ha is the horizontal alignment of the labels
    for i in range( len(rank_mean) ):
        plt.annotate('%1.2f' %np.round(rank_mean[i],2),xy=(x[i]+0.30,rank_mean[i]+0.30),fontsize='medium',rotation=90)
    
    xmin, xmax = plt.xlim()   # return the current xlim
    ymin, ymax = plt.ylim()   # return the current xlim
    plt.yticks(np.arange(1,11.1,1))
    plt.xlim([xmin-0.8,xmax+0.8]) # set the xlim 
    plt.ylim([ymin-0,ymax+0]) # set the xlim 
    plt.grid(axis='y')
    plt.legend(loc="upper left")
    plt.savefig('average_ranks.eps', format='eps', bbox_inches='tight') #bbox_inches='tight' to save legend outside the plot
    plt.tight_layout()
    plt.show() 

def plota_CD(rank, metodos, CD, pathFile='average_ranks.eps', formatFile='eps'):

    x = np.arange(0,len(metodos))
    rank_mean = np.mean(rank,0) #media do rank de cada algoritmo  

    asymmetric_error = [rank_mean-(rank_mean-CD),(rank_mean+CD)-rank_mean]
  
    plt.errorbar(x, rank_mean, yerr=asymmetric_error, linewidth=2, linestyle='None', marker='None', capsize=5, capthick=2, elinewidth=2, color='black', ecolor='red',  label="Critical difference", zorder=2)
    plt.scatter(x, rank_mean, marker='o', color='black', label="Average ranking", zorder=2) #zorder=1 faz com que o scatter fique na frente do errorbar 
    plt.xticks(x, metodos)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=25, ha='right',fontsize='large') #rotate x labels #ha is the horizontal alignment of the labels 
    for i in range( len(rank_mean) ):
        plt.annotate('%1.2f' %np.round(rank_mean[i],2),xy=(x[i]+0.05,rank_mean[i]+0.05),fontsize='large',rotation=90)
    
    xmin, xmax = plt.xlim()   # return the current xlim
    ymin, ymax = plt.ylim()   # return the current xlim
    plt.yticks(fontsize='large')
    #plt.yticks(np.arange(-1,8,1),fontsize='large')
    #plt.xlim([xmin-0.1,xmax+0.1]) # set the xlim 
    #plt.ylim([-0.6,6.5]) # set the xlim 
    plt.grid(axis='y')
    plt.ylabel(r'Average ranking',fontsize='large') 
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='large')

    #plt.legend(loc="upper left")
    plt.savefig(pathFile, format=formatFile, bbox_inches='tight') #bbox_inches='tight' to save legend outside the plot
    plt.tight_layout()
    plt.show()   

def plota_CD2(rank, metodos, CD, metodoInteresse = None, pathFile='average_ranks.eps', formatFile='eps'):

    if metodoInteresse is None:
        metodoInteresse = metodos[0]
        
    x = np.arange(0,len(metodos))
    #rank_mean = np.mean(rank,0) #media do rank de cada algoritmo  
    rank_mean = np.mean(rank,0) #media do rank de cada algoritmo  

    
    asymmetric_error = [rank_mean-(rank_mean-CD),(rank_mean+CD)-rank_mean]
  
    id_interesse = metodos.index(metodoInteresse)
    
    asymmetric_error[0] = rank_mean[id_interesse]
    asymmetric_error[1] = asymmetric_error[1][id_interesse]
    

    #if id_interesse==0:
    #    errorbar_min = 0;
    #else:
    errorbar_min = rank_mean[id_interesse]-(rank_mean[id_interesse]-CD);
                                 
    errorbar_max = (rank_mean[id_interesse]+CD)-rank_mean[id_interesse];
    
    plt.errorbar(x[id_interesse], rank_mean[id_interesse], yerr=[[errorbar_min],[errorbar_max]], linewidth=2, linestyle='None', marker='None', capsize=5, elinewidth=2, color='black', ecolor='red',  label="Critical difference", zorder=2)
    plt.scatter(x, rank_mean, marker='o', color='black', label="Average ranking", zorder=3) #zorder=1 faz com que o scatter fique na frente do errorbar 
    plt.xticks(x, metodos)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=25, ha='right') #rotate x labels #ha is the horizontal alignment of the labels 
    for i in range( len(rank_mean) ):
        plt.annotate('%1.2f' %np.round(rank_mean[i],2),xy=(x[i]+0.05,rank_mean[i]+0.05),fontsize='large',rotation=90)
    
    
    xmin, xmax = plt.xlim()   # return the current xlim
    ymin, ymax = plt.ylim()   # return the current xlim
    
    xmin, xmax = plt.xlim()   # return the current xlim
    #plt.xlim([xmin-0.1,xmax+0.1]) # set the xlim 
    #plt.ylim([1,ymax+1]) # set the xlim 
        
    xmin, xmax = plt.xlim()   # return the current xlim
    plt.plot([xmin, xmax],[rank_mean[id_interesse]+CD,rank_mean[id_interesse]+CD],color='red',linestyle='--', linewidth=1.0,zorder=1, label=None) #zorder=1 faz com que o scatter fique na frente do errorbar )
    
    if id_interesse>0:
        plt.plot([xmin, xmax],[rank_mean[id_interesse]-CD,rank_mean[id_interesse]-CD],color='red',linestyle='--', linewidth=1.0,zorder=1) #zorder=1 faz com que o scatter fique na frente do errorbar )
        
    #plt.yticks(np.arange(1,ymax+1,1))
    plt.yticks(fontsize='large')
    
    plt.grid(axis='y')
    plt.ylabel(r'Average ranking',fontsize='large') 
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    
    #plt.legend(loc="upper left")
    plt.savefig(pathFile, format=formatFile, bbox_inches='tight') #bbox_inches='tight' to save legend outside the plot
    plt.tight_layout()
    plt.show()  
