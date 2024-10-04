import pandas as pd
import numpy as np
from supportingFcn import toTime, haversine_distance, submit, stable_hash, convert_etaRaw_to_full_datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import tensorflow as tf

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


enginepower_mean = df_vessels['enginePower'].mean()
df_vessels.fillna({'enginePower': enginepower_mean}, inplace = True)
draft_mean = df_vessels['draft'].mean()
df_vessels.fillna({'draft': draft_mean}, inplace = True)

# Finding average speed when vessel is in operation
avg_sog_vessel = X_train[X_train['navstat'] == 0].groupby('vesselId')['sog'].mean().reset_index()
avg_sog_vessel.columns = ['vesselId', 'avg_sog']

# Enriching test/train dataset based on vesselId
X_train = X_train.merge(df_vessels[['vesselId', 'DWT', 'length', 'GT', 'yearBuilt']], on = 'vesselId', how = 'left')
X_test = X_test.merge(df_vessels[['vesselId', 'DWT', 'length', 'GT', 'yearBuilt']], on = 'vesselId', how = 'left')

X_train = X_train.merge(last_position[['vesselId','last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_position[['vesselId','last_latitude', 'last_longitude']], on = 'vesselId', how = 'left')

X_train = X_train.merge(last_sog[['vesselId', 'last_sog']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_sog[['vesselId', 'last_sog']], on = 'vesselId', how = 'left')

X_train = X_train.merge(last_heading[['vesselId', 'last_heading']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_heading[['vesselId', 'last_heading']], on = 'vesselId', how = 'left')

X_train = X_train.merge(last_day[['vesselId', 'last_day']], on = 'vesselId', how = 'left')
X_test = X_test.merge(last_day[['vesselId', 'last_day']], on = 'vesselId', how = 'left')

X_train = X_train.merge(avg_sog_vessel, on = 'vesselId', how = 'left')
X_test = X_test.merge(avg_sog_vessel, on = 'vesselId', how = 'left')

# Filling NaN's in datasets with mean sog
avg_sog_mean = avg_sog_vessel['avg_sog'].mean()
X_train.fillna({'avg_sog': avg_sog_mean}, inplace=True)
X_test.fillna({'avg_sog': avg_sog_mean}, inplace=True)

# Saving average delay per vesselId
df_arrival_times = X_train[['vesselId', 'time', 'navstat', 'etaRaw']].copy()
df_arrival_times.loc[:, 'time'] = pd.to_datetime(df_arrival_times['time'])
df_arrival_times.sort_values(by = ['vesselId', 'time'], inplace = True)
df_arrival_times['prev_navstat'] = df_arrival_times.groupby('vesselId')['navstat'].shift(1)
df_arrival_times.dropna(axis = 0, inplace = True)
df_arrival_times = df_arrival_times[(df_arrival_times['prev_navstat'] == 0) & (df_arrival_times['navstat'].isin([1, 5]))]
df_arrival_times.drop_duplicates(subset = ['vesselId'], keep = 'first', inplace = True)
df_arrival_times['etaRaw_parsed'] = df_arrival_times.apply(
    lambda row: convert_etaRaw_to_full_datetime(row['etaRaw'], row['time']), axis=1
)
df_arrival_times['arrival_deviation'] = (df_arrival_times['time'] - df_arrival_times['etaRaw_parsed']).dt.total_seconds() / 3600
df_avg_arrival_deviation = df_arrival_times.groupby('vesselId')['arrival_deviation'].mean().reset_index()
df_avg_arrival_deviation.rename(columns={'arrival_deviation': 'avg_arrival_deviation'}, inplace=True)
X_train = X_train.merge(df_avg_arrival_deviation, on = 'vesselId', how = 'left')
X_test = X_test.merge(df_avg_arrival_deviation, on = 'vesselId', how = 'left')

total_avg_arr_dev = df_avg_arrival_deviation['avg_arrival_deviation'].mean()
X_train.fillna({'avg_arrival_deviation': total_avg_arr_dev}, inplace = True)
X_test.fillna({'avg_arrival_deviation': total_avg_arr_dev}, inplace = True)

# Clustering the data based on geographical aspects
cluster_features_train = X_train[['latitude', 'longitude', 'avg_sog']].copy()
kmeans = KMeans(n_clusters = 3, random_state = 42)
X_train['cluster'] = kmeans.fit_predict(cluster_features_train)
vessel_cluster_mapping = X_train.groupby('vesselId')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
X_test = X_test.merge(vessel_cluster_mapping, on = 'vesselId', how = 'left')

# Encoding vesselId
X_train['vesselId_encoded'] = X_train['vesselId'].apply(stable_hash)
X_test['vesselId_encoded'] = X_test['vesselId'].apply(stable_hash)
# Dropping unused columns
X_train.drop(['vesselId', 'hour', 'minute', 'second', 'time', 'sog', 'navstat', 'etaRaw', 'portId', 
              'longitude', 'latitude', 'cluster', 'length', 'yearBuilt', 'heading'], axis = 1, inplace = True)
X_test.drop(['vesselId', 'hour', 'minute', 'second', 'time', 'cluster', 'length', 'yearBuilt'], axis = 1, inplace = True)

#print(X_test.columns)
#print(X_train.columns)

# NOTE: These are functions for running various tuned models.
def runModelforKaggle(X_train, X_test, y_lat, y_long):
    # Hyperparameters
    params = {'learning_rate': 0.08, 
              'max_depth': 10, 
              'min_child_weight': 35,
              'n_estimators': 200, 
              'reg_alpha': 0, 
              'reg_lambda': 0}
    # Initializing a XGBRegression model
    XGBlat = XGBRegressor()
    XGBlong = XGBRegressor()

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
        X, y_lat, y_long, test_size=0.2, random_state=42
    )
    # Hyperparameters
    params = {'learning_rate': 0.08, 
              'max_depth': 10, 
              'min_child_weight': 35,
              'n_estimators': 200, 
              'reg_alpha': 0, 
              'reg_lambda': 0}

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
    plt.show()

def runTFModelforTesting(X, y_lat, y_long):
    # Splitting the training data into a training and validation set
    X_train, X_val, y_lat_train, y_lat_val, y_long_train, y_long_val = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Scale the latitude target variable (y_lat)
    scaler_lat = StandardScaler()
    y_lat_train_scaled = scaler_lat.fit_transform(y_lat_train.values.reshape(-1, 1)).flatten()
    y_lat_val_scaled = scaler_lat.transform(y_lat_val.values.reshape(-1, 1)).flatten()

    # Scale the longitude target variable (y_long)
    scaler_long = StandardScaler()
    y_long_train_scaled = scaler_long.fit_transform(y_long_train.values.reshape(-1, 1)).flatten()
    y_long_val_scaled = scaler_long.transform(y_long_val.values.reshape(-1, 1)).flatten()

    # Combine scaled latitude and longitude for joint training
    y_train = np.column_stack((y_lat_train_scaled, y_long_train_scaled))
    y_val = np.column_stack((y_lat_val_scaled, y_long_val_scaled))

    # Building a TensorFlow model
    TFmodel = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation= 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(2)
    ])

    # Compiling a joint model
    TFmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    # Adding early stop
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fitting model for latitude and longitude
    TFmodel.fit(X_train_scaled, 
                y_train, 
                epochs=20, 
                validation_data=(X_val_scaled, y_val), 
                batch_size = 1024, 
                verbose=1, 
                callbacks = [early_stopping])

    # Predicting based on validation set
    y_pred = TFmodel.predict(X_val_scaled)

    # Separate latitude and longitude predictions
    y_lat_pred = y_pred[:, 0]
    y_long_pred = y_pred[:, 1]
    # Clipping as the model initially might predict values out of bounds
    y_lat_pred = np.clip(y_lat_pred, -90, 90)
    y_long_pred = np.clip(y_long_pred, -180, 180)

    # De-scale the predicted and true values before calculating RMSE and R²
    y_lat_pred_descaled = scaler_lat.inverse_transform(y_lat_pred.reshape(-1, 1)).flatten()
    y_long_pred_descaled = scaler_long.inverse_transform(y_long_pred.reshape(-1, 1)).flatten()

    y_lat_val_descaled = scaler_lat.inverse_transform(y_lat_val_scaled.reshape(-1, 1)).flatten()
    y_long_val_descaled = scaler_long.inverse_transform(y_long_val_scaled.reshape(-1, 1)).flatten()

    # Calculate RMSE for latitude and longitude
    rmse_lat = round(np.sqrt(root_mean_squared_error(y_lat_val_descaled, y_lat_pred_descaled)), 2)
    rmse_long = round(np.sqrt(root_mean_squared_error(y_long_val_descaled, y_long_pred_descaled)), 2)


    r2_lat = round(r2_score(y_lat_val_descaled, y_lat_pred_descaled), 2)
    r2_long = round(r2_score(y_long_val_descaled, y_long_pred_descaled), 2)

    # Calculate the Haversine distance
    #haversine_avg = haversine_distance(y_lat_val_descaled, y_long_val_descaled, y_lat_pred_descaled, y_long_pred_descaled)

    # Print evaluation results
    print(f'RMSE Latitude (descaled): {rmse_lat}')
    print(f'RMSE Longitude (descaled): {rmse_long}')
    print(f'R² Latitude (descaled): {r2_lat}')
    print(f'R² Longitude (descaled): {r2_long}')
    #print(f'Average Haversine Distance: {haversine_avg} km')
    # NOTE: Below plot can be used to show importance of each feature
    #plot_importance(XGBlat)
    #plt.show()

def runGridCV(X, y_lat, y_long):
    # Defining KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 10],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1]
    }

    # Initializing XGBRegressor models
    XGBlat = XGBRegressor(random_state=42, n_jobs=-1)
    XGBlong = XGBRegressor(random_state=42, n_jobs=-1)

    # Running GridSearchCV with KFold cross-validation for both latitude and longitude models
    grid_search_lat = GridSearchCV(estimator=XGBlat, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=1)
    grid_search_long = GridSearchCV(estimator=XGBlong, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=1)

    # Fitting grid search for latitude and longitude
    grid_search_lat.fit(X, y_lat)
    grid_search_long.fit(X, y_long)

    # Best hyperparameters from grid search
    print("Best hyperparameters for Latitude model:", grid_search_lat.best_params_)
    print("Best hyperparameters for Longitude model:", grid_search_long.best_params_)

    # Predicting on the best model
    y_lat_pred = grid_search_lat.best_estimator_.predict(X)
    y_lat_pred = np.clip(y_lat_pred, -90, 90)  # Clipping latitudes

    y_long_pred = grid_search_long.best_estimator_.predict(X)
    y_long_pred = np.clip(y_long_pred, -180, 180)  # Clipping longitudes

    # Calculate the Haversine distance
    haversine_avg = round(haversine_distance(y_lat, y_long, y_lat_pred, y_long_pred), 2)

    # Evaluate the model performance
    rmse_lat = round(root_mean_squared_error(y_lat, y_lat_pred), 2)
    rmse_long = round(root_mean_squared_error(y_long, y_long_pred), 2)

    # Calculate R² score
    r2_lat = round(r2_score(y_lat, y_lat_pred), 2)
    r2_long = round(r2_score(y_long, y_long_pred), 2)

    print(f'RMSE Latitude: {rmse_lat}')
    print(f'RMSE Longitude: {rmse_long}')
    print(f'R² Latitude: {r2_lat}')
    print(f'R² Longitude: {r2_long}')
    print(f'Average Haversine Distance: {haversine_avg} km')

    # Feature importance (Optional)
    # plot_importance(grid_search_lat.best_estimator_)
    # plt.show()

runModelforKaggle(X_train, X_test, y_lat, y_long)
#runXGBModelforTesting(X_train, y_lat, y_long)
#runTFModelforTesting(X_train, y_lat, y_long)
#runGridCV(X_train, y_lat, y_long)