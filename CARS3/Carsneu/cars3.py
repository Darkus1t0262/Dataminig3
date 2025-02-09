import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from flask import Flask, render_template, request

# ✅ 1. Cargar los archivos disponibles
def load_vehicle_data():
    files = ['autos_cabrera.csv', 'autos_confianza.csv', 'autos_1001carros.csv']
    
    df_list = []
    for file in files:
        if not os.path.exists(file):
            print(f"⚠️ Archivo no encontrado: {file}")
            continue
        
        try:
            df = pd.read_csv(file)
            
            column_mapping = {
                'Año': 'Year',
                'Precio': 'Precio',
                'Kilometraje': 'Kilometraje',
                'Marca': 'Marca',
                'Modelo': 'Modelo',
                'Transmisión': 'Transmision',
                'Dirección': 'Direccion',
                'Motor': 'Motor',
                'Tracción': 'Traccion',
                'Color': 'Color',
                'Combustible': 'Combustible'
            }
            df.rename(columns=column_mapping, inplace=True)

            required_columns = {'Modelo', 'Marca', 'Precio', 'Year', 'Kilometraje', 'Transmision', 'Direccion', 'Motor', 'Traccion', 'Color', 'Combustible'}
            
            if required_columns.issubset(df.columns):
                df = df[list(required_columns)].dropna()
                
                # Limpieza de Precio
                df['Precio'] = df['Precio'].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0').astype(float)
                
                # Limpieza de Kilometraje
                df['Kilometraje'] = df['Kilometraje'].astype(str).str.replace(r'[^\d]', '', regex=True).replace('', '0').astype(float)
                
                df_list.append(df)
            else:
                print(f"⚠️ Saltando {file}: Faltan columnas {required_columns - set(df.columns)}")
        except Exception as e:
            print(f"❌ Error cargando {file}: {e}")
    
    if not df_list:
        raise ValueError("❌ No se pudieron cargar datos válidos.")

    return pd.concat(df_list, ignore_index=True)

# ✅ 2. Preparar los datos para el modelo
def prepare_mlp_data(df):
    categorical_columns = ['Marca', 'Modelo', 'Transmision', 'Direccion', 'Motor', 'Traccion', 'Color', 'Combustible']
    numerical_columns = ['Year', 'Kilometraje']

    X = df[categorical_columns + numerical_columns].copy()
    y = df['Precio']

    # Filtrar valores extremos
    df = df[(df['Precio'] > 500) & (df['Precio'] < 200000)]  # Filtrar precios irreales
    df = df[(df['Kilometraje'] > 0) & (df['Kilometraje'] < 5000000)]  # Filtrar kilometrajes irreales

    # One-Hot Encoding para variables categóricas
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_columns])
    
    # Convertir DataFrame numérico en matriz
    X_numerical = X[numerical_columns].values

    # Concatenar variables categóricas con numéricas
    X_final = np.hstack((X_encoded, X_numerical))

    return X_final, y, df, encoder

# ✅ 3. Crear el modelo MLP
def build_mlp_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    df = load_vehicle_data()
    X, y, df, encoder = prepare_mlp_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ 4. Normalizar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ 5. Entrenar el modelo en CPU
    print("🔥 Entrenando modelo en CPU...")
    mlp_model = build_mlp_model(X_train.shape[1])
    history = mlp_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

    # ✅ 6. Guardar el modelo
    mlp_model.save("mlp_vehicle_model.keras")

    # ✅ 7. Generar y guardar el gráfico MAE
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Mean Absolute Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_plot.png')  # Guardar la imagen
    plt.close()

    # ✅ 8. Crear la aplicación Flask
    app = Flask(__name__)
    model = load_model("mlp_vehicle_model.keras")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Recibir los datos del formulario
            features = [
                request.form['marca'], request.form['modelo'], request.form['transmision'], request.form['direccion'],
                request.form['motor'], request.form['traccion'], request.form['color'], request.form['combustible'],
                int(request.form['year']), int(request.form['kilometraje'])
            ]

            # Convertir entrada a formato del modelo
            features_encoded = encoder.transform([features[:-2]])  # OneHotEncoding
            numerical_data = np.array(features[-2:], dtype=float).reshape(1, -1)  # Año y Kilometraje

            # Concatenar variables categóricas y numéricas
            input_data = np.hstack((features_encoded, numerical_data))
            input_data_scaled = scaler.transform(input_data)

            # Hacer predicción
            predicted_price = model.predict(input_data_scaled)[0][0]
            return render_template('result.html', predicted_price=round(predicted_price, 2))
        except Exception as e:
            return render_template('error.html', error=str(e))

    print("🚀 Iniciando servidor Flask en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
