#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:44:27 2020

@author: nico
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

# %%Iterations

RandomList = [random.randint(0,50) for i in range(3)]
Matrix = []

counter = 1
for j in RandomList:
    print("Iteration: "+str(counter))
    counter += 1
    
    for i in range(0,1,1):

        Data = pd.read_csv("dump.Al_slice.config", sep=" ", skiprows=9, index_col = False, names=['Id','Type','Pos_x','Pos_y','Pos_z','Svm','Epot','Cna','Vol'] )

        #Noisy data removal
        Data = Data.drop(Data.query('Cna == 2').sample(frac=1.0).index)
        Data = Data.drop(Data.query('Cna == 4').sample(frac=1.0).index) 

        #Label reassignment
        Data['Cna'] = Data['Cna'].replace(1,0)
        Data['Cna'] = Data['Cna'].replace(5,1)
    
        #Labels, Features, Scaling
        TrueLabels = Data['Cna']
        Features = Data.drop(columns = ['Id','Type','Pos_x','Pos_y','Pos_z', 'Cna'])
        sc = MinMaxScaler(feature_range=(0, 10))  
        sc.fit(Features) 
        Features_esc = pd.DataFrame(sc.transform(Features), index=Features.index)
        Features_esc.columns = ['Svm', 'Epot','Vol']
    
        #Kmeans algorithm
        clustering = KMeans(n_clusters = 2, init = 'k-means++', random_state = j, max_iter = 200, n_init = j+2, n_jobs = 4, algorithm = 'elkan')
        clusters = clustering.fit_predict(Features_esc)
    
        #Clusters addition to Features
        Features['Cluster'] = clusters
    
        #Clusters means
        mean_cluster = pd.DataFrame(round(Features.groupby('Cluster').mean(),3))
        print(mean_cluster)
        print(' ')
        print('Accuracy = '+str(round(accuracy_score(TrueLabels, Features['Cluster']), 3)))
        print('Precision = '+str(round(precision_score(TrueLabels, Features['Cluster'], average='binary', pos_label=0), 3)))
        print('Recall = '+str(round(recall_score(TrueLabels, Features['Cluster'], average='binary', pos_label=0), 3)))
        print('F1 = '+str(round(f1_score(TrueLabels, Features['Cluster'], average='weighted'), 4)))
        print('MCC = '+str(round(matthews_corrcoef(TrueLabels, Features['Cluster']), 4)))
        print(' ')

        Matrix.append(confusion_matrix(TrueLabels, Features['Cluster']))


# %%Confusion Matrix
mat0 = np.array(Matrix[0])
mat1 = np.array(Matrix[1])
mat2 = np.array(Matrix[2])

formatter = tkr.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-6, 6))

plt.figure(figsize = (18,4))

plt.subplot(1, 3, 1)    
sns.set(font_scale=1.3)
snsplot = sns.heatmap(mat0.T, square=False, annot=True, cbar=False, fmt='d', cmap="RdYlBu_r" , xticklabels = ['FCC','GB'], yticklabels = ['FCC','GB'])
snsplot.set(xlabel='True structure', ylabel='Predicted structure')
plt.text(-0.3,0.15, '(a)',fontsize=20, fontweight='bold')

plt.subplot(1, 3, 2)    
sns.set(font_scale=1.3)
snsplot = sns.heatmap(mat1.T, square=False, annot=True, cbar=False, fmt='d', cmap="RdYlBu_r" , xticklabels = ['FCC','GB'], yticklabels = ['FCC','GB'])
snsplot.set(xlabel='True structure', ylabel='Predicted structure')
plt.text(-0.3,0.15, '(b)',fontsize=20, fontweight='bold')

plt.subplot(1, 3, 3)    
sns.set(font_scale=1.3)
snsplot = sns.heatmap(mat2.T, square=False, annot=True, fmt='d', cmap="RdYlBu_r" , xticklabels = ['FCC','GB'], yticklabels = ['FCC','GB'], cbar_kws={'shrink': 1.0, 'label': 'Number of atoms', 'format': formatter})
snsplot.set(xlabel='True structure', ylabel='Predicted structure')
plt.text(-0.36,0.15, '(c)',fontsize=20, fontweight='bold')

snsplot.figure.savefig("2_Matrices.pdf", bbox_inches="tight")



