import pickle

import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/Users/Admin/Desktop/School/Short courses/Data Science/Project/Drug.csv")

df.drop(columns='Information', inplace=True)

df['Indication'] = df['Indication'].str.replace('\r\n', 'Unidentified')
df['Type'] = df['Type'].str.replace('\r\n', 'Unidentified')

df['Reviews'] = df['Reviews'].str.replace('Reviews', '')
df.rename(columns = {'Reviews':'Number of reviews'}, inplace = True)
df['Number of reviews'] = df['Number of reviews'].astype('float64')

effectiveness = []

for score in df['Effective']:
    if score < 1.0:
        effectiveness.append('Very Uneffective')
    elif score < 2.0:
        effectiveness.append('Uneffective')
    elif score < 3.0:
        effectiveness.append('Partly Effective')
    elif score < 4.0:
        effectiveness.append('More Than Effective')
    elif score <= 5.0:
        effectiveness.append('Very Effective')

df['level_of_effectiveness'] = effectiveness

easeofuse = []

for score in df['EaseOfUse']:
    if score < 1.0:
        easeofuse.append('Very Difficult')
    elif score < 2.0:
        easeofuse.append('Difficult')
    elif score < 3.0:
        easeofuse.append('Normal')
    elif score < 4.0:
        easeofuse.append('Easy')
    elif score <= 5.0:
        easeofuse.append('Very Easy')

df['level_of_difficulty'] = easeofuse

satisfaction_level = []

for score in df['Satisfaction']:
    if score < 1.0:
        satisfaction_level.append('Very Unsatisfied')
    elif score < 2.0:
        satisfaction_level.append('Unsatisfied')
    elif score < 3.0:
        satisfaction_level.append('Partly Satisfied')
    elif score < 4.0:
        satisfaction_level.append('More Than Satisfied')
    elif score <= 5.0:
        satisfaction_level.append('Very Satisfied')

df['level_of_satisfaction'] = satisfaction_level

df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True)



#Model
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=5)

#Transformation of data
train_df_selected=train_df[['Satisfaction','EaseOfUse','Effective','Number of reviews']]
test_df_selected=test_df[['Satisfaction','EaseOfUse','Effective','Number of reviews']]

# Perform log transform on the DataFrame
train_df_log = np.log(train_df_selected)
test_df_log = np.log(test_df_selected)
from sklearn.preprocessing import MinMaxScaler

# Perform normalization on the DataFrame
scaler = MinMaxScaler()
train_df_normalized = pd.DataFrame(scaler.fit_transform(train_df_selected), columns=train_df_selected.columns)
test_df_normalized = pd.DataFrame(scaler.fit_transform(test_df_selected), columns=test_df_selected.columns)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = train_df_normalized[['Number of reviews', 'EaseOfUse', 'Effective']]
y_train=train_df_normalized['Satisfaction']

X_test = test_df_normalized[['Number of reviews', 'EaseOfUse', 'Effective']]
y_test=test_df_normalized['Satisfaction']

# Initialize the linear regression model and fit it to the training data
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = model_reg.predict(X_test)

# Calculate the mean squared error of the model's predictions
mse = mean_squared_error(y_test, y_pred)


pickle.dump(model_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
