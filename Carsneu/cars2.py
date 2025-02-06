import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from flask import Flask, render_template, request

# Load vehicle data from uploaded CSV
def load_vehicle_data():
    file = 'resultados_1001carros.csv'

    df = pd.read_csv(file, sep=';')

    # Keep only relevant columns
    df = df[['Modelo', 'Precio', 'Year', 'Kilometraje']]
    df = df.dropna()

    # Clean and standardize the price column
    df['Precio'] = (
        df['Precio']
        .replace('[\$,]', '', regex=True)
        .replace(',', '.', regex=True)
        .astype(float)
    )

    return df

# Prepare data for MLP model
def prepare_mlp_data(df):
    X = df[['Modelo', 'Year', 'Kilometraje']].copy()
    y = df['Precio']

    # Convert text data in 'Modelo' to numerical features using Label Encoding
    le = LabelEncoder()
    X['Modelo'] = le.fit_transform(X['Modelo'].astype(str))

    return X, y, df, le

# Build MLP Model
def build_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main Workflow
df = load_vehicle_data()
X, y, df, le = prepare_mlp_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MLP Model on CPU
print("Training on CPU...")
mlp_model = build_mlp_model(X_train.shape[1])
history = mlp_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Save the model
mlp_model.save("mlp_vehicle_model.keras")

# Flask Web Application
app = Flask(__name__)

# Load the trained model
model = load_model("mlp_vehicle_model.keras")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        modelo = request.form['modelo']
        year = int(request.form['year'])
        kilometraje = int(request.form['kilometraje'])

        # Encode and scale input
        modelo_encoded = le.transform([modelo])[0]
        input_data = scaler.transform([[modelo_encoded, year, kilometraje]])

        # Predict price
        predicted_price = model.predict(input_data)[0][0]

        # Suggest platforms based on price
        if predicted_price < 10000:
            platform = "Consider checking low-cost platforms like OLX or Facebook Marketplace."
        elif predicted_price < 30000:
            platform = "You might find suitable options on platforms like AutoTrader or CarGurus."
        else:
            platform = "For premium vehicles, consider platforms like Cars.com or dealerships."

        return render_template('result.html', predicted_price=round(predicted_price, 2), platform=platform)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
