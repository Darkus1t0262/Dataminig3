import pandas as pd

file_paths = [
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

for file in file_paths:
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(file, sep=';')
        else:
            df = pd.read_excel(file)
        print(f"\n📌 Columns in {file}:")
        print(df.columns)
    except Exception as e:
        print(f"Error reading {file}: {e}")
