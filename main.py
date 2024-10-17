import pandas as pd
import numpy as np
from supportingFcn import toTime, haversine_distance, submit, stable_hash, convert_etaRaw_to_full_datetime
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder

# Load datasets
df_train = pd.read_csv('ais_train.csv', delimiter = '|')
df_test = pd.read_csv('ais_test.csv')
df_ports = pd.read_csv('ports.csv', delimiter = '|')
df_schedules = pd.read_csv('schedules_to_may_2024.csv', delimiter = '|')
df_vessels = pd.read_csv('vessels.csv', delimiter = '|')

featuresTrain = ['latitude', 'longitude', 'time', 'vesselId', 'sog', 'navstat', 'etaRaw', 'portId', 'heading']
featuresTest = ['time', 'vesselId']


# Selecting the features
X_train = df_train[featuresTrain]
X_test = df_test[featuresTest]
# Selecting the predictors
y_lat = df_train['latitude']
y_long = df_train['longitude']

print(X_train)

# Adding ratio terms
df_vessels['GT_length_ratio'] = df_vessels['GT'] / df_vessels['length']

# Splitting time into different features
X_train = toTime(X_train, 'measured', year = '2024')
X_test = toTime(X_test, 'measured', year = '2024')

# Adding last day from AIS data
last_day = X_train.groupby('vesselId').agg({
    'day': 'last'
}).reset_index()
last_day.rename(columns={
    'day': 'last_day'
}, inplace = True)

# Adding last known position of vessel
last_position = X_train.groupby('vesselId').agg({
    'latitude': 'last',
    'longitude': 'last'
}).reset_index()
last_position.rename(columns={
    'latitude': 'last_latitude',
    'longitude': 'last_longitude'
}, inplace=True)

# Adding last known sog
last_sog = X_train.groupby('vesselId').agg({
    'sog': 'last'
}).reset_index()
last_sog.rename(columns={
    'sog': 'last_sog'
}, inplace = True)

# Adding last known heading
last_heading = X_train.groupby('vesselId').agg({
    'heading': 'last'
}).reset_index()
last_heading.rename(columns={
    'heading': 'last_heading'
}, inplace = True)

label_encoder = LabelEncoder()
X_train['vesselId_encoded'] = label_encoder.fit_transform(X_train['vesselId'])
X_test['vesselId_encoded'] = label_encoder.transform(X_test['vesselId'])

# Enriching test/train dataset based on vesselId
X_train = X_train.merge(df_vessels[['vesselId', 'length', 'GT', 'yearBuilt']], on = 'vesselId', how = 'left')
X_test = X_test.merge(df_vessels[['vesselId', 'length', 'GT', 'yearBuilt']], on = 'vesselId', how = 'left')
print(X_train)

#X_train = X_train.merge(last_position[['vesselId','last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')
#X_test = X_test.merge(last_position[['vesselId','last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')

#X_train = X_train.merge(last_sog[['vesselId', 'last_sog']], on = 'vesselId', how = 'left')
#X_test = X_test.merge(last_sog[['vesselId', 'last_sog']], on = 'vesselId', how = 'left')

#X_train = X_train.merge(last_heading[['vesselId', 'last_heading']], on = 'vesselId', how = 'left')
#X_test = X_test.merge(last_heading[['vesselId', 'last_heading']], on = 'vesselId', how = 'left')


# Adding lagging terms
'''X_train['lag_1_lat'] = X_train.groupby('vesselId')['latitude'].shift(1)
X_train['lag_1_long'] = X_train.groupby('vesselId')['longitude'].shift(1)
X_train['lag_2_lat'] = X_train.groupby('vesselId')['latitude'].shift(2)
X_train['lag_2_long'] = X_train.groupby('vesselId')['longitude'].shift(2)
X_train['lag_1_heading'] = X_train.groupby('vesselId')['heading'].shift(1)
X_train['lag_2_heading'] = X_train.groupby('vesselId')['heading'].shift(2)

X_test['lag_1_lat'] = X_train.groupby('vesselId')['latitude'].shift(1)
X_test['lag_1_long'] = X_train.groupby('vesselId')['longitude'].shift(1)
X_test['lag_2_lat'] = X_train.groupby('vesselId')['latitude'].shift(2)
X_test['lag_2_long'] = X_train.groupby('vesselId')['longitude'].shift(2)
X_test['lag_1_heading'] = X_train.groupby('vesselId')['heading'].shift(1)
X_test['lag_2_heading'] = X_train.groupby('vesselId')['heading'].shift(2)'''

# Dropping unused columns
X_train.drop(['vesselId', 'hour', 'minute', 'second', 'time', 'sog', 'navstat', 'etaRaw', 'portId', 
              'longitude', 'latitude', 'length', 'yearBuilt', 
              'heading', 'month'], axis = 1, inplace = True)
X_test.drop(['vesselId', 'hour', 'minute', 'second', 'time', 'length', 'yearBuilt', 'month'], axis = 1, inplace = True)


# NOTE: These are functions for running various tuned models.
def runModelforKaggle(X_train, X_test, y_lat, y_long):

    # Initializing a XGBRegression model
    XGBlat = XGBRegressor(objective='reg:squarederror', n_estimators = 200, max_depth = 20, learning_rate = 1)
    XGBlong = XGBRegressor(objective='reg:squarederror', n_estimators = 200, max_depth = 20, learning_rate = 1)

    # Fitting model for latitude and longitude
    XGBlat.fit(X_train, y_lat)
    XGBlong.fit(X_train, y_long)

    # Predicting on test set
    y_lat_pred = XGBlat.predict(X_test)
    y_lat_pred = np.clip(y_lat_pred, -90, 90) # Clipping as the model initially might predict values out of bounds
    y_long_pred = XGBlong.predict(X_test)
    y_long_pred = np.clip(y_long_pred, -180, 180) # Clipping as the model initially might predict values out of bounds
    
    # Creating a CSV output file
    submit(df_test['ID'], y_long_pred, y_lat_pred)

def runXGBModelforTesting(X, y_lat, y_long):
    # Splitting the training data into a training and validation set
    X_train, X_val, y_lat_train, y_lat_val, y_long_train, y_long_val = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42, shuffle = False
    )
    # Hyperparameters
    params = {
        'learning_rate': 0.01,          # Smaller learning rate for more gradual learning
        'max_depth': 10,                # Max depth of the trees (higher values for more complexity)
        'min_child_weight': 5,          # Minimum sum of instance weights (control overfitting)
        'n_estimators': 600,            # More trees can work well with a lower learning rate
        'reg_alpha': 0.1,               # L1 regularization (helps with feature selection)
        'reg_lambda': 1.0,              # L2 regularization (helps prevent overfitting)
        'subsample': 0.8,               # Randomly sample a fraction of data (prevent overfitting)
        'colsample_bytree': 0.8,        # Randomly sample features for each tree
        'gamma': 0,                     # Minimum loss reduction for a split, can be increased to make trees simpler
        'objective': 'reg:squarederror', # Objective for regression
    }
    # Initializing a XGBRegression model
    XGBlat = XGBRegressor()
    XGBlong = XGBRegressor()

    # Fitting model for latitude and longitude
    XGBlat.fit(X_train, y_lat_train)
    XGBlong.fit(X_train, y_long_train)

    # Predicting on test set
    y_lat_pred = XGBlat.predict(X_val)
    y_lat_pred = np.clip(y_lat_pred, -90, 90) # Clipping as the model initially might predict values out of bounds
    y_long_pred = XGBlong.predict(X_val)
    y_long_pred = np.clip(y_long_pred, -180, 180) # Clipping as the model initially might predict values out of bounds

    # Calculate the Haversine distance
    haversine_avg = round(haversine_distance(y_lat_val, y_long_val, y_lat_pred, y_long_pred), 2)

    # Evaluate the model performance
    rmse_lat = round(root_mean_squared_error(y_lat_val, y_lat_pred), 2)
    rmse_long = round(root_mean_squared_error(y_long_val, y_long_pred), 2)
    # Calculate R² score
    r2_lat = round(r2_score(y_lat_val, y_lat_pred), 2)
    r2_long = round(r2_score(y_long_val, y_long_pred), 2)

    print(f'RMSE Latitude: {rmse_lat}')
    print(f'RMSE Longitude: {rmse_long}')
    print(f'R² Latitude: {r2_lat}')
    print(f'R² Longitude: {r2_long}')
    print(f'Average Haversine Distance: {haversine_avg} km')
    # NOTE: Below plot can be used to show importance of each feature
    plot_importance(XGBlat)
    plot_importance(XGBlong)
    #plt.show()

#runModelforKaggle(X_train, X_test, y_lat, y_long)
#runXGBModelforTesting(X_train, y_lat, y_long)