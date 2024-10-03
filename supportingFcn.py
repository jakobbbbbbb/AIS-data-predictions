import pandas as pd
from haversine import haversine
import hashlib
from geopy.distance import geodesic

def toTime(df, timetype, year='2024'):
    df = df.copy()
    if timetype == 'measured':
        # Convert and extract features from measured time
        df['time'] = pd.to_datetime(df['time'], errors = 'coerce')
        df.loc[:, 'hour'] = df['time'].dt.hour
        df.loc[:, 'day'] = df['time'].dt.day
        df.loc[:, 'minute'] = df['time'].dt.minute
        df.loc[:, 'second'] = df['time'].dt.second
        df.loc[:, 'month'] = df['time'].dt.month
        #df.drop('time', axis=1, inplace=True)

    elif timetype == 'eta':
        # Handle missing or erroneous etaRaw values
        df['etaRaw'] = df['etaRaw'].fillna('01-01 00:00')  # Default placeholder if missing
        df['etaRaw'] = f'{year}-' + df['etaRaw']  # Add the year dynamically
        df['etaRaw'] = pd.to_datetime(df['etaRaw'], errors='coerce')  # Convert with error handling
        
        # Extract ETA-related features
        df['hour_Eta'] = df['etaRaw'].dt.hour
        df['dayofweek_Eta'] = df['etaRaw'].dt.dayofweek
        df['day_Eta'] = df['etaRaw'].dt.day
        df['minute_Eta'] = df['etaRaw'].dt.minute
        df['second_Eta'] = df['etaRaw'].dt.second
        df.drop('etaRaw', axis=1, inplace=True)
    
    return df

# Define a threshold distance (in km) for when a vessel is considered to have arrived at a port
def is_near_port(row, port_lat, port_long, threshold_km=5):
    vessel_location = (row['latitude_vessel'], row['longitude_vessel'])
    port_location = (port_lat, port_long)
    distance = geodesic(vessel_location, port_location).kilometers
    return distance < threshold_km

# Create a DataFrame for submission
def submit(test_ids, ylong, ylat):
    # Create a submission DataFrame
    submission = pd.DataFrame({
        'ID': test_ids,  # Pass the test ID column
        'longitude_predicted': ylong,  # Your predicted longitudes
        'latitude_predicted': ylat    # Your predicted latitudes
    })

    # Ensure the correct column order
    submission = submission[['ID', 'longitude_predicted', 'latitude_predicted']]

    # Save to CSV
    submission.to_csv('jakobs_results.csv', index=False)


# Function for calculating the haversine distance between vessel and port
def calculate_haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0  # If any value is NaN, return 0 distance
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def haversine_distance(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    total_distance = 0
    n = len(y_true_lat)
    
    for lat_true, lon_true, lat_pred, lon_pred in zip(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
        # Calculate distance between true and predicted lat/lon
        point_true = (lat_true, lon_true)
        point_pred = (lat_pred, lon_pred)
        distance = haversine(point_true, point_pred)
        total_distance += distance
    
    # Return the average Haversine distance
    return total_distance / n

# Function for setting a stable seed for hashing
def stable_hash(x):
    return int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16) % 10**6

