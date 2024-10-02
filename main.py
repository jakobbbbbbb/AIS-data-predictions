import pandas as pd
import numpy as np
from supportingFcn import toTime, haversine_distance, submit, stable_hash, is_near_port
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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


featuresTrain = ['time', 'vesselId']
featuresTest = ['time', 'vesselId']

# Selecting the features
X_train = df_train[featuresTrain]
X_test = df_test[featuresTest]
# Selecting the predictors
y_lat = df_train['latitude_vessel']
y_long = df_train['longitude_vessel']


# Splitting time into different features
X_train = toTime(X_train, 'measured', year = '2024')
X_test = toTime(X_test, 'measured', year = '2024')


# Filling the NaNs for DWT with the mean value and grouping by size
DWT_mean = df_vessels['DWT'].mean()
df_vessels.fillna({'DWT': DWT_mean}, inplace = True)
df_vessels['DWT_grouped'] = pd.qcut(df_vessels['DWT'], q = 5, labels = ['Very small', 'Small', 'Medium', 'Large', 'Very large'])
DWT_encoder = LabelEncoder()
df_vessels['DWT_grouped_encoded'] = DWT_encoder.fit_transform(df_vessels['DWT_grouped'])

# Filling the NaNs for length with the mean value and grouping by size
length_mean = df_vessels['length'].mean()
df_vessels.fillna({'length': length_mean}, inplace = True)
df_vessels['length_grouped'] = pd.qcut(df_vessels['length'], q = 5, labels = ['Very short', 'Short', 'Medium', 'Long', 'Very long'])
length_encoder = LabelEncoder()
df_vessels['length_grouped_encoded'] = length_encoder.fit_transform(df_vessels['length_grouped'])

# Enriching test/train dataset based on vesselId
X_train = X_train.merge(df_vessels[['vesselId', 'DWT_grouped_encoded', 'length_grouped_encoded', 'GT']], on = 'vesselId', how = 'left')
X_test = X_test.merge(df_vessels[['vesselId', 'DWT_grouped_encoded', 'length_grouped_encoded', 'GT']], on = 'vesselId', how = 'left')

# Encoding vesselId
X_train['vesselId_encoded'] = X_train['vesselId'].apply(stable_hash)
X_test['vesselId_encoded'] = X_test['vesselId'].apply(stable_hash)
X_train.drop(['vesselId', 'hour', 'minute', 'second', 'time'], axis = 1, inplace = True)
X_test.drop(['vesselId', 'hour', 'minute', 'second', 'time'], axis = 1, inplace = True)


def runModelforKaggle(X_train, X_test, y_lat, y_long):
    params = {
        'n_estimators': 30,
        'learning_rate': 1,
        'max_depth': 50,
        'random_state': 42,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'n_jobs': -1,
    }

    # Initializing a XGBRegression model
    XGBlat = XGBRegressor(**params)
    XGBlong = XGBRegressor(**params)

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
    params = {
        'n_estimators': 30,
        'learning_rate': 1,
        'max_depth': 50,
        'random_state': 42,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'n_jobs': -1,
    }

    # Initializing a XGBRegression model
    XGBlat = XGBRegressor(**params)
    XGBlong = XGBRegressor(**params)

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
    #plot_importance(XGBlat)
    #plot_importance(XGBlong)
    #plt.show()

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
    # Splitting the training data into a training and validation set
    X_train, X_val, y_lat_train, y_lat_val, y_long_train, y_long_val = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [10, 20, 30],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [20, 50, 80]
    }

    # Initializing XGBRegressor models
    XGBlat = XGBRegressor(random_state=42, n_jobs=-1)
    XGBlong = XGBRegressor(random_state=42, n_jobs=-1)

    # Running GridSearchCV for both latitude and longitude models
    grid_search_lat = GridSearchCV(estimator=XGBlat, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search_long = GridSearchCV(estimator=XGBlong, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

    # Fitting grid search for latitude and longitude
    grid_search_lat.fit(X_train, y_lat_train)
    grid_search_long.fit(X_train, y_long_train)

    # Best hyperparameters from grid search
    print("Best hyperparameters for Latitude model:", grid_search_lat.best_params_)
    print("Best hyperparameters for Longitude model:", grid_search_long.best_params_)

    # Predicting on validation set using the best model
    y_lat_pred = grid_search_lat.best_estimator_.predict(X_val)
    y_lat_pred = np.clip(y_lat_pred, -90, 90) # Clipping latitudes

    y_long_pred = grid_search_long.best_estimator_.predict(X_val)
    y_long_pred = np.clip(y_long_pred, -180, 180) # Clipping longitudes

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

    # Feature importance (Optional)
    #plot_importance(grid_search_lat.best_estimator_)
    #plt.show()

runModelforKaggle(X_train, X_test, y_lat, y_long)
#runXGBModelforTesting(X_train, y_lat, y_long)
#runTFModelforTesting(X_train, y_lat, y_long)
#runGridCV(X_train, y_lat, y_long)