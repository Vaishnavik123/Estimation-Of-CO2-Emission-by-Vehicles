import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.model_selection import train_test_split

model = pk.load(open('my_model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))
encoder = pk.load(open('encoder.pkl', 'rb'))

def bold_text(text):
    return f"**{text}**"

st.header('Smoke prediction model')

smoke_df = pd.read_csv('CO2 Emissions_Canada.csv')

smoke_df.rename(columns={
    'Make': 'make', 'Model': 'model', 'Engine Size(L)': 'engine_size', 'Cylinders': 'cylinders',
    'Transmission': 'transmission', 'Fuel Consumption City (L/100 km)': 'fuel_consumption_city',
    'Fuel Consumption Hwy (L/100 km)': 'fuel_consumption_hwy', 'Fuel Consumption Comb (L/100 km)': 'fuel_consumption_comb_l',
    'Fuel Type': 'fuel_type', 'Vehicle Class': 'vehicle_class', 'Fuel Consumption Comb (mpg)': 'fuel_consumption_mpg',
    'CO2 Emissions(g/km)': 'co2_emissions'}, inplace=True)

smoke_df.drop(['fuel_consumption_comb_l', 'cylinders', 'model', 'transmission','fuel_consumption_mpg'], axis='columns', inplace=True)


make = st.selectbox(bold_text('Select Model Name'), smoke_df['make'].unique())
vehicle_class = st.selectbox(bold_text('Select Vehicle Class'), smoke_df['vehicle_class'].unique())
engine_size = st.slider(bold_text('Engine size in L'), 1.0, 10.0)
fuel_type = st.selectbox(bold_text('Fuel Type'), smoke_df['fuel_type'].unique())
fuel_consumption_city = st.slider(bold_text('Consumption of fuel in city in L/100 km'), 4.0, 30.0)
fuel_consumption_hwy = st.slider(bold_text('Consumption of fuel on highway in L/100 km'), 4.0, 25.0)


# Train-test split
train_df, test_df = train_test_split(smoke_df, test_size=0.30, train_size=0.70, random_state=42)

# Extract input columns and target column
input_cols = smoke_df.columns.tolist()[0:-1]
target_col = 'co2_emissions'

# Define train and test datasets
train_inputs = train_df[input_cols]
train_target = train_df[target_col]

test_inputs = test_df[input_cols]
test_target = test_df[target_col]

numeric_cols = smoke_df.select_dtypes('number').columns.tolist()[0:-1]
object_cols = smoke_df.select_dtypes('object').columns.tolist()

encoded_cols = encoder.get_feature_names_out(object_cols).tolist()

# Prediction function
def predict(make, vehicle_class, engine_size, fuel_type, fuel_consumption_city, fuel_consumption_hwy):
    data = {
        'make': [make],
        'vehicle_class': [vehicle_class],
        'engine_size': [engine_size],
        'fuel_type': [fuel_type],
        'fuel_consumption_city': [fuel_consumption_city],
        'fuel_consumption_hwy': [fuel_consumption_hwy],

    }
    df = pd.DataFrame(data)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[encoded_cols] = encoder.transform(df[object_cols])
    X_t = df[numeric_cols + encoded_cols]
    predict_me = model.predict(X_t)
    return predict_me

# Trigger prediction and display the result
if st.button("Predict"):
    prediction = predict(make, vehicle_class, engine_size, fuel_type, fuel_consumption_city, fuel_consumption_hwy)
    formatted_prediction = f'{prediction[0]:.2f}'
    st.markdown(bold_text(f'The Smoke emitted is {formatted_prediction} gm/Km'), unsafe_allow_html=True)


# Display maximum and minimum values in the 'co2_emissions' column
if 'co2_emissions' in smoke_df.columns:
    max_value = smoke_df['co2_emissions'].max()
    min_value = smoke_df['co2_emissions'].min()
    st.markdown(bold_text(f"Maximum value in emission recorded is: {max_value} gm/Km"), unsafe_allow_html=True)
    st.markdown(bold_text(f"Minimum value in emission recorded is: {min_value} gm/Km"), unsafe_allow_html=True)
