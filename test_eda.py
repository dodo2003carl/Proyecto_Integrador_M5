import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuración de visualización
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Cargar datos
data_path = 'Base_de_datos.xlsx'
if os.path.exists(data_path):
    df = pd.read_excel(data_path)
    print(f"Datos cargados: {df.shape}")
else:
    print(f"No se encontró el archivo en {data_path}")
    exit(1)

# 1. Análisis Univariable
# Distribución de variable objetivo 'Pago_atiempo'
print("\n--- Análisis Univariable ---")
print(df['Pago_atiempo'].value_counts(normalize=True))

# Variables Numéricas Importantes
num_cols = ['capital_prestado', 'edad_cliente', 'puntaje', 'salario_cliente']

# Just checking if code runs without errors, not showing plots
for col in num_cols:
    if col in df.columns:
        print(f"Processing {col}...")
        try:
            sns.histplot(df[col], kde=True)
            plt.close()
            sns.boxplot(x=df[col])
            plt.close()
        except Exception as e:
            print(f"Error plotting {col}: {e}")

# Variables Categóricas
if 'tipo_laboral' in df.columns:
    print("Processing tipo_laboral...")
    try:
        sns.countplot(y='tipo_laboral', data=df)
        plt.close()
    except Exception as e:
        print(f"Error plotting tipo_laboral: {e}")

# 2. Análisis Bivariable
print("\n--- Análisis Bivariable ---")
for col in num_cols:
    if col in df.columns:
        try:
            sns.boxplot(x='Pago_atiempo', y=col, data=df)
            plt.close()
        except Exception as e:
             print(f"Error plotting bivariate {col}: {e}")

# 3. Análisis Multivariable
print("\n--- Análisis Multivariable ---")
try:
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr_matrix)
    plt.close()
except Exception as e:
    print(f"Error plotting correlation matrix: {e}")

print("EDA script completed successfully.")
