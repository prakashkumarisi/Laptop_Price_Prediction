import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the model and DataFrame
pipe = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
company_encoded = encoder.fit_transform(df[['Company']]).toarray()
type_encoded = encoder.transform(df[['TypeName']]).toarray()
cpu_encoded = encoder.transform(df[['Cpu brand']]).toarray()
gpu_encoded = encoder.transform(df[['Gpu brand']]).toarray()
os_encoded = encoder.transform(df[['os']]).toarray()

# Concatenate all encoded features
encoded_features = np.concatenate((company_encoded, type_encoded, cpu_encoded, gpu_encoded, os_encoded), axis=1)

st.title("Laptop Predictor")
st.write("Made by Prakash Kumar")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert categorical variables to one-hot encoded format
    company_index = np.where(df['Company'].unique() == company)[0][0]
    type_index = np.where(df['TypeName'].unique() == type)[0][0]
    cpu_index = np.where(df['Cpu brand'].unique() == cpu)[0][0]
    gpu_index = np.where(df['Gpu brand'].unique() == gpu)[0][0]
    os_index = np.where(df['os'].unique() == os)[0][0]
    
    query_encoded = np.zeros((1, encoded_features.shape[1]))
    query_encoded[0, company_index] = 1
    query_encoded[0, type_index + len(df['Company'].unique())] = 1
    query_encoded[0, cpu_index + len(df['Company'].unique()) + len(df['TypeName'].unique())] = 1
    query_encoded[0, gpu_index + len(df['Company'].unique()) + len(df['TypeName'].unique()) + len(df['Cpu brand'].unique())] = 1
    query_encoded[0, os_index + len(df['Company'].unique()) + len(df['TypeName'].unique()) + len(df['Cpu brand'].unique()) + len(df['Gpu brand'].unique())] = 1
    
    # Other numerical features
    touchscreen_encoded = 1 if touchscreen == 'Yes' else 0
    ips_encoded = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    
    # Concatenate all features
    query = np.concatenate(([ram, weight, touchscreen_encoded, ips_encoded, ppi, hdd, ssd], query_encoded), axis=None)
    query = query.reshape(1, -1)
    
    # Predict price
    predicted_price = np.exp(pipe.predict(query)[0])
    st.title("The predicted price of this configuration is $" + str(int(predicted_price)))
