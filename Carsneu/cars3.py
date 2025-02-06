import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from flask import Flask, render_template, request

# Load vehicle data from multiple uploaded CSV/XLSX files
def load_vehicle_data():
    files = [
        'resultados_1001carros.csv',
        'productos_mercado-libre.xlsx',
        'productos_mercado-libre2.xlsx',
        'autos_autocosmos.xlsx',
        'autos_autocosmos2.xlsx',
        'autos_autocosmos3.xlsx',
        'autos_autocosmos4.xlsx',
        'autos_autocosmos5.xlsx',
        'autos_autocosmos6.xlsx'
    ]
    
    df_list = []
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file, sep=';')
            else:
                df = pd.read_excel(file)
            
            # Standardize column names
            column_mapping = {
                'Año': 'Year',
                'Precio': 'Precio',
                'Kilometraje': 'Kilometraje',
                'Nombre': 'Modelo'  # Adjust Mercado Libre format
            }
            df.rename(columns=column_mapping, inplace=True)
            
            required_columns = {'Modelo', 'Marca', 'Precio', 'Year', 'Kilometraje'}
            
            if required_columns.issubset(df.columns):
                df = df[['Modelo', 'Marca', 'Precio', 'Year', 'Kilometraje']].dropna()
                
                # Clean "Precio" column (remove 'us', '$', ',', convert to float)
                df['Precio'] = (
                    df['Precio']
                    .astype(str)
                    .str.replace(r'[^\d.]', '', regex=True)
                    .astype(float)
                )
                
                # Clean "Kilometraje" column (remove "km" and convert to numeric)
                df['Kilometraje'] = df['Kilometraje'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)
                
                df_list.append(df)
            else:
                print(f"Skipping {file}: Missing required columns {required_columns - set(df.columns)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not df_list:
        raise ValueError("No valid data files loaded.")
    
    return pd.concat(df_list, ignore_index=True)

# Prepare data for MLP model
def prepare_mlp_data(df):
    X = df[['Modelo', 'Marca', 'Year', 'Kilometraje']].copy()
    y = df['Precio']
    
    # Convert text data to numerical features using Label Encoding
    le_modelo = LabelEncoder()
    le_marca = LabelEncoder()
    X['Modelo'] = le_modelo.fit_transform(X['Modelo'].astype(str))
    X['Marca'] = le_marca.fit_transform(X['Marca'].astype(str))
    
    return X, y, df, le_modelo, le_marca

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
X, y, df, le_modelo, le_marca = prepare_mlp_data(df)
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

# Generate and save MAE plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('mae_plot.png')  # Save plot to file
plt.show()

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
        marca = request.form['marca']
        year = int(request.form['year'])
        kilometraje = int(request.form['kilometraje'])

        # Encode and scale input
        marca_encoded = le_marca.transform([marca])[0]
        input_data = scaler.transform([[marca_encoded, year, kilometraje]])

        # Predict price
        predicted_price = model.predict(input_data)[0][0]

        # Suggest platforms based on price
        if predicted_price < 10000:
            platform = "Consider checking low-cost platforms like OLX or Facebook Marketplace."
            link = "https://listado.mercadolibre.com.ec/venta-de-autos-usados-quito_Frenos-ABS_Si"
        elif predicted_price < 30000:
            platform = "You might find suitable options on platforms like AutoTrader or CarGurus."
            link = "https://www.autocosmos.com.ec/auto/usado?pidx=6"
        else:
            platform = "For premium vehicles, consider platforms like Cars.com or dealerships."
            link = "https://www.autocosmos.com.ec/auto/usado?pidx=6"

        return render_template('result.html', predicted_price=round(predicted_price, 2), platform=platform, link=link)
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
