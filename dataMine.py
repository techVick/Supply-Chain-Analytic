# -*- coding: utf-8 -*-
"""
Vikulp Sharma
Data Mining TEXL
"""
import pandas as pd
import os                                                                                                             
import csv

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r   
p=list_files('C:/Users/307006397/Desktop/TEXL_TAT/Anil_Mails/xlsx/texlupdate')
with open('C:/Users/307006397/Desktop/TEXL_TAT/Anil_Mails/xlsx/906344.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    for i in range(43):
        xls = pd.ExcelFile(p[i])
        df1 = xls.parse('Engines')
        row=df1.shape[0]
        for j in range(row):
            ff=df1.index[j]
            dd=ff[1]
            if dd==906344:
               writer.writerows([[p[i]],[j],[df1.iloc[j]]])

p=list_files('C:/Users/307006397/Desktop/TEXL_TAT/Anil_Mails/xlsx/criticalparts')
for i in range(43):
    xls = pd.ExcelFile(p[i])
    df1 = xls.parse('Tracker')
    df1.rename(columns={'Data current as of:':'aa','Unnamed: 1': 'a','Unnamed: 2': 'b','Unnamed: 4': 'd','Unnamed: 5': 'e','Unnamed: 6': 'f','Unnamed: 7': 'g','Unnamed: 8': 'h','Unnamed: 9': 'i','Unnamed: 10': 'j','Unnamed: 11': 'k','Unnamed: 12': 'l','Unnamed: 13': 'm','Unnamed: 14': 'n','Unnamed: 15': 'o','Unnamed: 16': 'p','Unnamed: 17': 'q','Unnamed: 18': 'r'}, inplace=True)
    df1=df1.loc[df1['aa'] == 906344]
    df1.to_csv("ff.csv", mode='a',columns=["aa","a", "b", "c","d", "e", "f","g", "h", "i","j", "k", "l","m", "n", "o","p", "q", "r"])
