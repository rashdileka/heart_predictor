# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset (Make sure heart.csv is in same folder)
dataset = pd.read_csv('heart.csv')

X = dataset.drop('target', axis=1)
y = dataset['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (takes some time)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

def predict_heart_disease(input_data):
    # input_data: list of 13 features (same order as dataset columns except target)
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return int(prediction[0][0] > 0.5)
