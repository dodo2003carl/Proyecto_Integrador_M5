import pandas as pd
import sys

file_path = 'Base_de_datos.xlsx'
output_file = 'inspect_data_output_utf8.txt'

try:
    df = pd.read_excel(file_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Columns: {df.columns.tolist()}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Dtypes:\n{df.dtypes}\n")
        f.write(f"Head:\n{df.head()}\n")
    print(f"Successfully wrote to {output_file}")
except Exception as e:
    print(f"Error reading excel: {e}")
