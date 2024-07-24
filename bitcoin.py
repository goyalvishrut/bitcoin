from numpy.lib.type_check import imag
import pandas as pd
from scipy.sparse.linalg import lsqr
# from sklearn import svm
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from PIL import Image
from sklearn.metrics import mean_squared_error as mse
import streamlit as st
import math

st.write("""
# Bitcoin 
# Predict Bitcoin price using Machine Learning
""")

# Get the data
df = pd.read_csv('bitcoin_price_Training.csv')

#managing the data
df['Volume']=df['Volume'].str.replace(",","")
df['Volume']=df['Volume'].str.replace("-","0")
df['Market Cap']=df['Market Cap'].str.replace(",","")
df['Market Cap']=df['Market Cap'].str.replace("-","0")
df['Volume']=pd.to_numeric(df['Volume'])
df['Market Cap']=pd.to_numeric(df['Market Cap'])
df['Date']=pd.to_datetime(df['Date'],format="%b %d, %Y").dt.date

#set a subheader
st.subheader('Data Information')

#show the data frame
st.dataframe(df)

# show statistics on the data
st.write(df.describe())

# show correlation value
st.write(df.corr())

#show data as chart
bar = st.bar_chart(df.iloc[:,1:8])

#removing the date column
date = df['Date']
df.drop('Date',axis=1,inplace=True)

#split the data into independent 'x' and dependent 'y' variable

x= df.loc[:, ["Open","High","Low","Volume","Market Cap"]].values
y=df.loc[:,["Close"]].values

#split the data set into 75% training and 25% testing

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=1)

#get the feature input from the user
#Date,Open,High,Low,Close,Volume,Market Cap
def get_user_input():
    # date = st.sidebar.slider('Date',0,5000,2000)
    open = st.sidebar.slider('Open',0,1000,2000)
    high = st.sidebar.slider('High',0,10000,5000)
    low = st.sidebar.slider('Low',0,10000,1000)
    volume = st.sidebar.slider('Volume',math.ceil(df['Volume'].min()/1000000),math.ceil(df['Volume'].max()/1000000),math.ceil(int(df['Volume'].mean()/1000000)))
    market_cap = st.sidebar.slider('Market capital',math.ceil(df['Market Cap'].min()/1000000),math.ceil(df['Market Cap'].max()/1000000),math.ceil(int(df['Market Cap'].mean()/1000000)))
    

    #store a dictionary into a variable
    user_data = {
                # 'date': date,
                 'open':open,
                 'high': high,
                 'low': low,
                 'volume': volume,
                 'market_cap': market_cap
                }

    #transfer the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#store the user input variable
user_input = get_user_input()

#set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

#randomforest

#create and train the model for Randomforest
model_RandomForestRegressor = RandomForestRegressor(criterion='mse')
model_RandomForestRegressor.fit(x_train,y_train)

#show the model metrices
st.subheader('Model Test MSE Score for RandomForest: ')
st.write( str(mse(y_test, model_RandomForestRegressor.predict(x_test))))

#store the model prediction in a variable
prediction_randomforest = model_RandomForestRegressor.predict(user_input)

#set a subheader and display the classification
st.subheader('regression  from random forest: ')
st.write(prediction_randomforest)


#decisiontree

#create and train the model for Decisiontress
model_DecisionTreeRegressor = DecisionTreeRegressor(criterion='mse')
model_DecisionTreeRegressor.fit(x_train,y_train)

st.subheader('Model Test MSE Score for DecisionTree: ')
st.write( str(mse(y_test, model_DecisionTreeRegressor.predict(x_test))))

#store the model prediction in a variable
prediction_decisiontree = model_DecisionTreeRegressor.predict(user_input)

#set a subheader and display the classification
st.subheader('regression  from decision tree: ')
st.write(prediction_decisiontree)



#svr

#create and train the model for Decisiontress
model_SVR = SVR()
model_SVR.fit(x_train,y_train)

st.subheader('Model Test MSE Score for SVR: ')
st.write( str(mse(y_test, model_SVR.predict(x_test))))

#store the model prediction in a variable
prediction_decisiontree = model_SVR.predict(user_input)

#set a subheader and display the classification
st.subheader('regression  from SVR: ')
st.write(prediction_decisiontree)