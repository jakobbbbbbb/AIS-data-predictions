import pandas as pd
import numpy as np
from supportingFcn import toTime, haversine_distance, submit
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from tpot import TPOTRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load datasets
df_train = pd.read_csv('ais_train.csv', delimiter = '|')
df_test = pd.read_csv('ais_test.csv')
df_ports = pd.read_csv('ports.csv', delimiter = '|')
df_schedules = pd.read_csv('schedules_to_may_2024.csv', delimiter = '|')
df_vessels = pd.read_csv('vessels.csv', delimiter = '|')

featuresTrain = ['latitude', 'longitude', 'time', 'vesselId', 'navstat']
featuresTest = ['time', 'vesselId']

X_train = df_train[featuresTrain]
X_test = df_test[featuresTest]

y_lat = df_train['latitude']
y_long = df_train['longitude']

X_train = toTime(X_train, 'measured', year = '2024')
X_test = toTime(X_test, 'measured', year = '2024')

# Adding last known position of vessel
last_position = X_train.groupby('vesselId').agg({
    'latitude': 'last',
    'longitude': 'last'
}).reset_index()
last_position.rename(columns={
    'latitude': 'last_latitude',
    'longitude': 'last_longitude'
}, inplace=True)
# Merging last position in test/train datasets 
X_train = X_train.merge(last_position[['vesselId', 'last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_position[['vesselId', 'last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')

# Adding last known navstat
last_navstat = X_train.groupby('vesselId').agg({
    'navstat': 'last',
}).reset_index()
last_navstat.rename(columns={
    'navstat': 'last_navstat'
}, inplace = True)

X_train = X_train.merge(last_navstat[['vesselId', 'last_navstat']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_navstat[['vesselId', 'last_navstat']], on = 'vesselId', how = 'left')

X_train = X_train.merge(df_vessels[['vesselId', 'length', 'yearBuilt', 'CEU', 'GT']], on = 'vesselId', how = 'left')
X_test = X_test.merge(df_vessels[['vesselId', 'length', 'yearBuilt']], on = 'vesselId', how = 'left')

X_train.drop(['time', 'vesselId', 'latitude', 'longitude', 'navstat', 'second', 'hour', 'minute'], axis = 1, inplace = True)
X_test.drop(['time', 'vesselId'], axis = 1, inplace = True)

# Creating X_train_lat and X_train_long as some features don't make sense to include in lat/long models
X_train_lat = X_train.drop(['last_longitude', 'day'], axis = 1, inplace = False)
X_train_long = X_train.drop(['last_latitude', 'yearBuilt', 'day', 'CEU'], axis = 1, inplace = False)

# Attempting to find the best features based on correlation
corrlat = X_train_lat.corrwith(y_lat).sort_values(ascending = False)
plt.figure(figsize=(14,8))
plt.bar(corrlat.index, corrlat.values)
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.title('Correlation with latitude')

corrlong = X_train_long.corrwith(y_long).sort_values(ascending = False)
plt.figure(figsize=(14,8))
plt.bar(corrlong.index, corrlong.values)
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.title('Correlation with longitude')
print(X_train_lat.columns)
#plt.show()
#print(df_vessels.isna().any())

# Implementing a RandomForestRegressor
def runRFR(X_lat, X_long, y_lat, y_long):
    # Splitting data for latitude prediction
    X_train_lat, X_val_lat, y_lat_train, y_lat_val = train_test_split(
        X_lat, y_lat, test_size=0.2, random_state=42
    )

    # Splitting data for longitude prediction
    X_train_long, X_val_long, y_long_train, y_long_val = train_test_split(
        X_long, y_long, test_size=0.2, random_state=42
    )

    # Initializing a XGBRegression model
    RFRlat = RandomForestRegressor(n_jobs = -1, verbose = 2, random_state = 42)
    RFRlong = RandomForestRegressor(n_jobs = -1, verbose = 2, random_state = 42)

    # Fitting model for latitude and longitude
    RFRlat.fit(X_train_lat, y_lat_train)
    RFRlong.fit(X_train_long, y_long_train)

    # Predicting on test set
    y_lat_pred = RFRlat.predict(X_val_lat)
    y_lat_pred = np.clip(y_lat_pred, -90, 90) # Clipping as the model initially might predict values out of bounds
    y_long_pred = RFRlong.predict(X_val_long)
    y_long_pred = np.clip(y_long_pred, -180, 180) # Clipping as the model initially might predict values out of bounds

    # Calculate the Haversine distance
    haversine_avg = round(haversine_distance(y_lat_val, y_long_val, y_lat_pred, y_long_pred), 2)

    # Evaluate the model performance
    rmse_lat = round(root_mean_squared_error(y_lat_val, y_lat_pred), 2)
    rmse_long = round(root_mean_squared_error(y_long_val, y_long_pred), 2)
    # Calculate R² score
    r2_lat = round(r2_score(y_lat_val, y_lat_pred), 2)
    r2_long = round(r2_score(y_long_val, y_long_pred), 2)

    #cv_scores_lat = cross_val_score(RFRlat, X_train, y_lat_train, cv=5, scoring='neg_mean_squared_error')
    #cv_scores_long = cross_val_score(RFRlong, X_train, y_long_train, cv=5, scoring='neg_mean_squared_error')

    #print("Latitude CV MSE: ", -cv_scores_lat.mean())
    #print("Longitude CV MSE: ", -cv_scores_long.mean())

    print(f'RMSE Latitude: {rmse_lat}')
    print(f'RMSE Longitude: {rmse_long}')
    print(f'R² Latitude: {r2_lat}')
    print(f'R² Longitude: {r2_long}')
    print(f'Average Haversine Distance: {haversine_avg} km')

def runKaggle(X_train, X_test, y_lat, y_long):
    # Initializing a XGBRegression model
    RFRlat = RandomForestRegressor(n_jobs = -1, verbose = 2, random_state = 42)
    RFRlong = RandomForestRegressor(n_jobs = -1, verbose = 2, random_state = 42)

    # Fitting model for latitude and longitude
    RFRlat.fit(X_train, y_lat)
    RFRlong.fit(X_train, y_long)

    # Predicting on test set
    y_lat_pred = RFRlat.predict(X_test)
    y_lat_pred = np.clip(y_lat_pred, -90, 90) # Clipping as the model initially might predict values out of bounds
    y_long_pred = RFRlong.predict(X_test)
    y_long_pred = np.clip(y_long_pred, -180, 180) # Clipping as the model initially might predict values out of bounds
    
    # Creating a CSV output file
    submit(df_test['ID'], y_long_pred, y_lat_pred)

#runRFR(X_train_lat, X_train_long, y_lat, y_long)
#runKaggle(X_train, X_test, y_lat, y_long)