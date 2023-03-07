
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Simple Advertising App

""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tv_length = st.sidebar.slider('TV', 0, 400, 147)
    radio_length = st.sidebar.slider('Radio', 0, 50, 23)
    Newspaper_length = st.sidebar.slider('Newspaper', 0, 144, 30)
    
    data = {'TV': tv_length,
            'Radio': radio_length,
            'Newspaper': Newspaper_length}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

raw_df = pd.read_csv('/content/Advertising.csv')

st.subheader('User Input parameters')
st.write(df)

X = raw_df[['TV','Radio','Newspaper']]
Y = raw_df['Sales']

clf = RandomForestRegressor()
clf.fit(X, Y)

prediction = clf.predict(df)
print(prediction)
st.subheader('Prediction')
st.write(prediction[0])
