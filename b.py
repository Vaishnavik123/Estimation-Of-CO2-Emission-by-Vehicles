import opendatasets as od
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
import pickle as pk


smoke_df=pd.read_csv('CO2 Emissions_Canada.csv')

smoke_df.rename(columns={'Make':'make','Model':'model','Engine Size(L)':'engine_size','Cylinders':'cylinders','Transmission':'transmission','Fuel Consumption City (L/100 km)':'fuel_consumption_city','Fuel Consumption Hwy (L/100 km)':'fuel_consumption_hwy','Fuel Consumption Comb (L/100 km)':'fuel_consumption_comb_l','Fuel Type':'fuel_type','Vehicle Class':'vehicle_class','Fuel Consumption Comb (mpg)':'fuel_consumption_mpg','CO2 Emissions(g/km)':'co2_emissions'},inplace=True)

smoke_df.drop(['fuel_consumption_comb_l'],axis='columns',inplace=True)
smoke_df.drop(['fuel_consumption_mpg'],axis='columns',inplace=True)
smoke_df.drop(['model'],axis='columns',inplace=True)
smoke_df.drop(['transmission'],axis='columns',inplace=True)
smoke_df.drop(['cylinders'],axis='columns',inplace=True)




train_df,test_df=train_test_split(smoke_df,test_size=0.30,train_size=0.70,random_state=42)

input_cols=smoke_df.columns.tolist()[0:-1]
target_col='co2_emissions'

train_inputs=train_df[input_cols]
train_target=train_df[target_col]

test_inputs=test_df[input_cols]
test_target=test_df[target_col]

numeric_cols=smoke_df.select_dtypes('number').columns.tolist()[0:-1]
object_cols=smoke_df.select_dtypes('object').columns.tolist()

encoder=OneHotEncoder(sparse_output=False)
encoder.fit(smoke_df[object_cols])

encoded_cols=encoder.get_feature_names_out(object_cols).tolist()

train_inputs.loc[:,encoded_cols]=encoder.transform(train_inputs[object_cols])
test_inputs.loc[:,encoded_cols]=encoder.transform(test_inputs[object_cols])

scaler=MinMaxScaler()
scaler.fit(smoke_df[numeric_cols])

train_inputs.loc[:,numeric_cols]=scaler.transform(train_inputs[numeric_cols])
test_inputs.loc[:,numeric_cols]=scaler.transform(test_inputs[numeric_cols])

X_train=train_inputs[numeric_cols+encoded_cols]
X_test=test_inputs[numeric_cols+encoded_cols]


lasso_model=LassoCV(cv=5,random_state=42)
lasso_model.fit(X_train,train_target)


def predict(make, vehicle_class, engine_size, fuel_type, fuel_consumption_city, fuel_consumption_hwy,
            fuel_consumption_mpg):
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
    prediction = lasso_model.predict(X_t)
    return prediction
print(lasso_model.score(X_test,test_target))


def rmse(actual, prediction):
    return np.sqrt(np.mean(np.square(prediction - actual)))
print(rmse(test_target,lasso_model.predict(X_test)))

pk.dump(lasso_model,open('my_model.pkl',"wb"))
pk.dump(encoder,open('encoder.pkl',"wb"))
pk.dump(scaler,open('scaler.pkl',"wb"))
