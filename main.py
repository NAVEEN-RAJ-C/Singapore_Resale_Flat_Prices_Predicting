# import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# read the model ready csv file
df = pd.read_csv('c.csv')
def rf_model(size, df):
    df = df.sample(n=size, random_state= 42)
    # declare the varialbes to provide to ML model
    X = df.drop('resale_price', axis = 1)
    y = df['resale_price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train,y_train)


# X_train and y_train will be your training data, and X_test and y_test will be your testing data

def main():
    st.set_page_config(page_title='Singapore Flat Resale Price Predictor', layout='wide',initial_sidebar_state='expanded')
    st.sidebar.title('Please provides the required details')
    st.title('Singapore Flat Resale Price Predictor')
    st.header('Please enter the sample size')
    sample_size = st.number_input('sample size', value=5000)
    if st.button('train the model'):
        rf_model(sample_size, df)


if __name__ == '__main__':
    main()