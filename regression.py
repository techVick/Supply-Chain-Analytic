# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:34:46 2017
Vikulp Sharma
TEXL TAT Reduction
"""
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime, date
import seaborn as sns
from collections import Counter
from pandas import Series
from pylab import figure, axes, pie, title, show
import csv
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
import csv
import sys
# reading salesforce DR report, SV Visit for GEnx product line
texlTat=pd.read_csv("C:/Users/307006397/Desktop/TEXL_TAT/tattat.csv")
texlTat.rename(columns={'M53 (HPC)':'hpc'}, inplace=True)
#texlTatMod=texlTat[texlTat.hpc.str.contains("UNK") == False]
texlTatMod=texlTat.loc[texlTat['Site'] == 'TEXL\n']
ESN=texlTatMod['ESN']
seed1=pd.read_csv("C:/Users/307006397/Desktop/TEXL_TAT/seed.csv")
seed1['EventDate'] = pd.to_datetime(seed1['EventDate'])
seed2=seed1
out_csv = 'C:/Users/307006397/Desktop/TEXL_TAT/vik5.csv'
with open('vik1.csv','a') as f1:
    writer=csv.writer(f1, delimiter='\t',)
    csv_new = csv.writer(open('C:/Users/307006397/Desktop/F&B_Priority/DR_Analytics/textt.csv', 'w'))
    for i in ESN:
        seed1=seed1.loc[seed1['ESN'] == i] ## Update
        seed1=seed1.loc[seed1['EventDate'] >= '2015-01-01 00:00:00'] ## Update
        seed1=seed1.loc[seed1['EventDate'] <= '2017-01-01 00:00:00']
        ii=seed1['ESN']
        bb=seed1['EventDate']
        jj=seed1['EventType']
        kk=seed1['SVCategory']
        ll=seed1['RemovalCount']
        mm=seed1['FDMSVScheduleIndicator']
        nn=seed1['FDMRemScheduleIndicator']
        frames=[ii,bb,jj,kk,ll,mm,nn]
        result = pd.concat(frames,axis=1)
#        result.columns=["ATASubtask","caseOwner","ATASubtask","Title","ATASubtask","endineModel","ATASubtask","caseOwner","ATASubtask","partNo","ATASubtask","PartName"]
#        writer.writerow([ii,jj,kk,ll,mm,nn])
#        csv_new.writerow(i)
        result.to_csv(out_csv,index=False,header=False,mode='a')#size of data to append for each loop
        seed1=seed2

#
#texlTatMod=texlTatMod.loc[texlTatMod['Site'] == 'Wales']
#texlTatMod=texlTatMod.loc[texlTatMod['hpc'] == '4']
#print(texlTatMod.describe())



#### Bar Char Plotting
#kk=texlTatMod.groupby('hpc')['ESN'].count()
#kk1=texlTatMod.groupby('M59 (HPTM)')['ESN'].count()
#kk2=texlTatMod.groupby('M60 (LPTM)')['ESN'].count()
#kk3=texlTatMod.groupby('M45 (CDN)')['ESN'].count()
#kk4=texlTatMod.groupby('M56 (S2N)')['ESN'].count()
#kk5=texlTatMod.groupby('M57 (HPTR)')['ESN'].count()
#kk6=texlTatMod.groupby('M58 (TCF)')['ESN'].count()
#kk7=texlTatMod.groupby('M61 (LPTR)')['ESN'].count()
#kk8=texlTatMod.groupby('M63 (TRF)')['ESN'].count()
#kk9=texlTatMod.groupby('M64(FMS)')['ESN'].count()
#
#frames=[kk]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='HPC Wales WorkScope Levels', align='center');## Update
#
#frames=[kk1]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='HPTM Wales WorkScope Levels', align='center');#
#
#frames=[kk2]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='LPTM Wales WorkScope Levels', align='center');#
#
#frames=[kk3]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='CDN Wales WorkScope Levels', align='center');#
#
#frames=[kk4]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='S2N Wales WorkScope Levels', align='center');#
#
#frames=[kk5]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='HPTR Wales WorkScope Levels', align='center');#
#
#frames=[kk6]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='TCF Wales WorkScope Levels', align='center');#
#
#frames=[kk7]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='LPTR Wales WorkScope Levels', align='center');#
#
#frames=[kk8]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='TRF Wales WorkScope Levels', align='center');#
#frames=[kk9]
#result = pd.concat(frames,axis=1)
#result.plot(kind='bar',title='FMS Wales WorkScope Levels', align='center');#  
       
            
            
#######   LOGISTIC REGRESSION PLOTTING            
#logistic = LogisticRegression()
#
#X=texlTatMod[['hpc','M59 (HPTM)','M60 (LPTM)','M45 (CDN)','M56 (S2N)','M57 (HPTR)','M58 (TCF)','M61 (LPTR)','M63 (TRF)','M64(FMS)']]
#X1=texlTatMod[['G1-3','hpc']]
#X2=texlTatMod[['G1-3','M59 (HPTM)']]
#X3=texlTatMod[['G1-3','M60 (LPTM)']]
#X4=texlTatMod[['G1-3','M45 (CDN)']]
#X5=texlTatMod[['G1-3','M56 (S2N)']]
#X6=texlTatMod[['G1-3','M57 (HPTR)']]
#X7=texlTatMod[['G1-3','M58 (TCF)']]
#X8=texlTatMod[['G1-3','M61 (LPTR)']]
#X9=texlTatMod[['G1-3','M63 (TRF)']]
#X10=texlTatMod[['G1-3','M64(FMS)']]
#Y1=texlTatMod[['G1-3']]
#Y11=texlTatMod[['Gate 2','Gate 3']]
#Y2=texlTatMod[['G0+Ship']]
#Y3=texlTatMod[['Gate 0']]
#Y4=texlTatMod[['Gate 1']]
#Y5=texlTatMod[['Gate 2']]
#Y6=texlTatMod[['Gate 3']]
#Y7=texlTatMod[['Gate 4']]
#lor=logistic.fit(X,Y1)
#result=lor.coef_
#result1=lor.n_iter_
#g = sns.FacetGrid(X10)
#g.map(sns.boxplot,'M64(FMS)', 'G1-3')
#print(Y11.describe)
#mdl = sm.MNLogit(X1.astype(int),Y1)
#mdl_fit = mdl.fit()
#print(mdl_fit.summary())




#mdl = sm.MNLogit(X2.astype(int),Y1)
#mdl_fit = mdl.fit()
#print(mdl_fit.summary())
##print lor.summary()
#X2 = sm.add_constant(X)
#est = sm.OLS(Y1, X2.astype(float))
#est2 = est.fit()
#print(est2.summary())
#cov = np.cov(X1.astype(float), rowvar=False)
#print(cov)