from numpy.lib.type_check import imag
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from PIL import Image
from sklearn.metrics import mean_squared_error as mse
import streamlit as st

st.write("""
# Bitcoin 
""")

# Get the data
df = pd.read_csv('C:/Users/vishrut.goyal/Desktop/webapp/bitcoin/bitcoin_price_Training.csv')

#managing the data
df['Volume']=df['Volume'].str.replace(",","")
df['Volume']=df['Volume'].str.replace("-","0")
df['Market Cap']=df['Market Cap'].str.replace(",","")
df['Market Cap']=df['Market Cap'].str.replace("-","0")
df['Volume']=pd.to_numeric(df['Volume'])
df['Market Cap']=pd.to_numeric(df['Market Cap'])
df['Date']=pd.to_datetime(df['Date'],format="%b %d, %Y").dt.date

#show the data
st.write(df)

st.write(df.corr())