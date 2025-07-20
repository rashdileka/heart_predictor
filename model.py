import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset (Make sure heart.csv is in the same folder)
dataset = pd.read_csv('heart.csv')

# Separate features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Save feature names for future use
feature_names = X.columns.tolist()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)


def predict_heart_disease(input_data):
    """
    input_data: list of 13 values in correct order
    Returns: 1 if heart disease is predicted, else 0
    """
    # Convert input list to DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale input using trained scaler
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction_prob = model.predict(input_scaled)
    prediction = int(prediction_prob[0][0] > 0.5)
    return prediction
