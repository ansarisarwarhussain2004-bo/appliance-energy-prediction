import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel("energydata_complete.xlsx")

# Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month

df.drop('date', axis=1, inplace=True)

# Define X and y
X = df.drop('Appliances', axis=1)
y = df['Appliances']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

print("Model trained successfully!")