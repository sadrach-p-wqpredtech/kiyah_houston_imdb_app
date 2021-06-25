#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:47:57 2021

@author: sadrachpierre
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



st.write("""
# IMDB Movie Success Prediction App


""")

st.sidebar.header('User Input Features')


df_selected = pd.read_csv("titles_m_update.csv")
print(df_selected.head())

from collections import Counter
# print(dict(Counter((df_selected['genres'])).most_common(50)).keys())
# df_selected = df_selected[df_selected['genres'].isin(['Drama', 'Documentary', 'Comedy', 'Thriller', 'Action', 
#                                                       'Romance', 'Western', 'Family', 'Crime', 'Adventure', 'Sci-Fi'])]

# df_selected.to_csv("titles_m_update.csv", index = False)
#'Drama', 'Documentary', 'Comedy', 'Thriller', 'Action', 'Romance', 'Western', 'Family', 'Crime', 'Adventure', 'Sci-Fi'



df_selected_all = df_selected[['genres', 'runtimeMinutes', 'startYear', 'averageRating']].copy()

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="imdb_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    st.write("""Correlation heatmaps are a great way to visualize, not only the relationship betwen input variables, but also the relationship
     between our inputs and our target. This can help with identifying which input features most strongly influence an outcome. In our heatmap
     we see that there is a relatively strong correlation between the start year and average rating.""")
    df_selected_all.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, cmap="Blues", annot=True)
    st.pyplot()
    


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        genres = st.sidebar.selectbox('genres',('Drama', 'Documentary', 'Comedy', 'Thriller', 
                                                'Action', 'Romance', 'Western', 'Family', 'Crime', 'Adventure', 'Sci-Fi'))       
        startYear = st.sidebar.slider('Start Year', 1931.0,2019.0, 1931.0)
        runtimeMinutes = st.sidebar.slider('Runtime in Minutes', 3.0,300.0, 20.0)

        data = {'genres':[genres], 
                'startYear':[startYear], 
                'runtimeMinutes':[runtimeMinutes], 
                }
        
        features = pd.DataFrame(data)
        return features
    input_df = user_input_features()


imdb_raw = pd.read_csv('titles_m_update.csv')




imdb_raw.fillna(0, inplace=True)
averageRating = imdb_raw.drop(columns=['averageRating'])
df = pd.concat([input_df,imdb_raw],axis=0)




encode = ['genres']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)


print(df.columns)


features = ['startYear', 'runtimeMinutes', 'genres_Action',
       'genres_Adventure', 'genres_Comedy', 'genres_Crime',
       'genres_Documentary', 'genres_Drama', 'genres_Family', 'genres_Romance',
       'genres_Sci-Fi', 'genres_Thriller', 'genres_Western']

df = df[features]



# Displays the user input features
st.subheader('User Input features')
print(df.columns)
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_reg = pickle.load(open('imdb_reg.pkl', 'rb'))

# Apply model to make predictions
prediction = load_reg.predict(df)



st.subheader('Prediction')

st.write("Predicted Rating: ", prediction)

