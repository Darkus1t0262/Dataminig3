# 1. Price Prediction and Catalog Generation Using MLP and Generative AI for Vehicles
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from fpdf import FPDF
from flask import Flask, render_template, send_from_directory

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

    return X, y, df

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

# Create a PDF Catalog
class CatalogPDF(FPDF):
    def header(self):
        self.set_font("Arial", size=12)
        self.cell(0, 10, "Vehicle Catalog", ln=True, align="C")

def create_pdf_catalog(df):
    pdf = CatalogPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for i, row in df.iterrows():
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Vehicle {i+1}", ln=True)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Model: {row['Modelo']}\nYear: {row['Year']}\nMileage: {row['Kilometraje']}\nPrice: ${row['Precio']}")

    pdf.output("Vehicle_Catalog.pdf")

# Main Workflow
df = load_vehicle_data()
X, y, df = prepare_mlp_data(df)
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

# Evaluate the model
test_loss, test_mae = mlp_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Create the PDF catalog
create_pdf_catalog(df)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Flask Web Application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', vehicles=df.to_dict(orient='records'))

@app.route('/catalog')
def catalog():
    return send_from_directory(directory=os.getcwd(), path="Vehicle_Catalog.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
