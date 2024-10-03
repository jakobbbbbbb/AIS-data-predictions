import matplotlib.pyplot as plt
import pandas as pd


# Load datasets
df_train = pd.read_csv('ais_train.csv', delimiter = '|')
df_test = pd.read_csv('ais_test.csv')
df_ports = pd.read_csv('ports.csv', delimiter = '|')
df_schedules = pd.read_csv('schedules_to_may_2024.csv', delimiter = '|')
df_vessels = pd.read_csv('vessels.csv', delimiter = '|')

#plt.hist(df_train['sog'], bins = 100)
plt.hist(df_vessels['DWT'], bins = 100)
print(df_vessels['yearBuilt'].min())
#plt.show()
