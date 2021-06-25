#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:44:59 2021

@author: sadrachpierre
"""
import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_imdb = pd.read_csv('titles_m_update.csv')
df_imdb = df_imdb[['genres', 'runtimeMinutes', 'startYear', 'averageRating']].copy()

print(df_imdb.head())

print(df_imdb['runtimeMinutes'].min())
print(df_imdb['runtimeMinutes'].max())

print(df_imdb['startYear'].min())
print(df_imdb['startYear'].max())




df = df_imdb.copy()
df.fillna(0, inplace=True)


encode = ['genres']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]



# Separating X and y

X = df.drop('averageRating', axis=1)
Y = df['averageRating']



print(X.columns)
# Build random forest model

reg = LinearRegression()
reg.fit(X, Y)

# Saving the model

pickle.dump(reg, open('imdb_reg.pkl', 'wb'))
