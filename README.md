
# Estimation-of-C02-Emission-by-vehicles
Deploy a ML model to calculate the CO2 Emission by vehicles

The working of the project can be viewed from link given below:
https://estimation-of-c02-emission-by-vehicles-nh9qh7vn3yrltysdpcqsim.streamlit.app/

Dataset:
It is almost a cleaned dataset with no missing values
This dataset is taken From kaggle which is issued by a Cannda Government.
This dataset contains 7385 rows and 12 columns.
The detailed description along with a link to download it is given below:
https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles


ML model:
The code provided has a detailed analysis using matplotlib and seaborn libraries.
Some of the attributes for which the target has very less dependency has been removed to
avoid unnecessary complications.
First the model is evaluated with a linear regression model but the model overfits the data with
the accuracy of 99%.
Hence the overcome the overfitting issues Lasso model has been used with the accuracy of 93%
and resolving the overfitting issue.
The detailed explanation is provided in the code section.

Libraries:
pandas, matplotlib, seaborn,opendatasets,sklearn and numpy

Streamlit:
This model is deployed using Streamlit web application.
Pycharm is used where the code for the model is inherited from jupyter notebook
and collaborated in pycharm by using pickle module.
Each of the Constructer is transfered from model.py to app.py using pkl extensions.
To create files of pkl extensions the code is written in model.py.
When you run the model.py in pycharm the pkl extension files for each constructor will be created automatically.
Manually, you need add all those pickle files into github and create requirements.txt file to deploy an app.
.
