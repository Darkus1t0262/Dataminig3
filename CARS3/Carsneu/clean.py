import pandas as pd
import os

# List of AutoCosmos files
autocosmos_files = [
    "autos_autocosmos.xlsx",
    "autos_autocosmos2.xlsx",
    "autos_autocosmos3.xlsx",
    "autos_autocosmos4.xlsx",
    "autos_autocosmos5.xlsx",
    "autos_autocosmos6.xlsx"
]

def clean_autocosmos_data(files):
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue
        
        try:
            df = pd.read_excel(file)

            if 'Precio' in df.columns:
                # Convert to string, remove 'u$s', extra spaces, and replace ',' with '.'
                df['Precio'] = (
                    df['Precio']
                    .astype(str)
                    .str.strip()
                    .str.replace(r'[^\d,.]', '', regex=True)  # Remove non-numeric characters
                    .str.replace(',', '.', regex=True)  # Convert to standard decimal
                )

                # Convert to float (ignoring errors)
                df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')

                # Save cleaned file
                cleaned_filename = f"cleaned_{file}"
                df.to_excel(cleaned_filename, index=False)
                print(f"Successfully cleaned {file} -> {cleaned_filename}")

            else:
                print(f"Skipping {file}: 'Precio' column not found")

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Run the cleaning function
clean_autocosmos_data(autocosmos_files)
