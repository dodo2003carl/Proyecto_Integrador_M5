
"""
Componente de Ingeniería de Características.
Este script carga los datos, realiza la partición estratificada y aplica
transformaciones (imputación, encoding) diferenciadas por tipo de variable.
Retorna los conjuntos de entrenamiento y prueba procesados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer

def cargar_datos(ruta_archivo):
    """
    Carga los datos desde un archivo Excel o CSV.
    """
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")
    
    if ruta.suffix == '.xlsx':
        df = pd.read_excel(ruta)
    elif ruta.suffix == '.csv':
        df = pd.read_csv(ruta)
    else:
        raise ValueError("Formato de archivo no soportado. Use .xlsx o .csv")
    
    return df

def feature_engineering(df, target_col='Pago_atiempo'):
    """
    Aplica el pipeline de ingeniería de características.
    Retorna X_train_proc, X_test_proc, y_train, y_test y el preprocesador ajustado.
    """
    
    # 1. Separación de Variables (X) y Target (y)
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no se encuentra en el dataframe.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Eliminar columnas que no se usarán o son identificadores/fechas complejas para este alcance
    # 'fecha_prestamo' se podría usar para extraer mes/año, pero por simplicidad inicial la excluimos o tratamos numéricamente si convertimos.
    # Vamos a excluirla para evitar errores simples, o convertirla a timestamp. Mejor excluirla por ahora.
    cols_to_drop = ['fecha_prestamo']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    # 2. Definición de Tipos de Variables
    # Basado en el análisis EDA y df.info()
    
    # Numéricas: float64 e int64 (excepto categoicas ya conocidas)
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Categóricas Nominales: object/string
    # Categóricas Nominales: object/string (excluyendo ordinales si se separan)
    # Por defecto todo lo string. Si separamos 'tendencia_ingresos':
    nominal_cols = [c for c in X.select_dtypes(include=['object', 'string']).columns if c != 'tendencia_ingresos']
    ordinal_cols = ['tendencia_ingresos'] if 'tendencia_ingresos' in X.columns else []
    
    categorical_cols = nominal_cols + ordinal_cols # Para debug
    
    print(f"Variables Numéricas ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables Nominales ({len(nominal_cols)}): {nominal_cols}")
    print(f"Variables Ordinales ({len(ordinal_cols)}): {ordinal_cols}")

    
    # Pre-procesamiento: Asegurar tipos explícitamente para evitar errores de OneHotEncoder con int/str
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    # IMPORTANTE: Reemplazar 'nan' string (generado por astype) por np.nan para que SimpleImputer funcione
    X.replace('nan', np.nan, inplace=True) 


    # 3. División Train/Test Estratificada
    # Es crucial estratificar por el target si hay desbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Construcción de Pipelines
    
    # Pipeline Numérico: Imputación (Mediana) + Escalado
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline Categórico (Nominal): Imputación (Constante/Moda) + OneHot
    # Es importante asegurar que todo sea string para evitar error "mixed types"
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Pre-procesamiento: Asegurar tipos
    X[categorical_cols] = X[categorical_cols].astype(str)
    
    # Pipeline Ordinal: Imputación + OrdinalEncoder
    # 'tendencia_ingresos'
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # O constante
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('nom', categorical_transformer, nominal_cols),
            ('ord', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop' # Elimina columnas no especificadas
    )
    
    # 5. Ajuste y Transformación
    # Ajustamos solo con TRAIN para evitar Data Leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Recuperar nombres de columnas para facilitar interpretación (Opcional pero recomendado Senior)
    # Intento de obtener nombres de features
    try:
        num_names = numeric_cols
        
        # Nombres de nominales (OneHot)
        nom_names = preprocessor.named_transformers_['nom']['onehot'].get_feature_names_out(nominal_cols)
        
        # Nombres de ordinales (Ordinal) - mantienen el nombre original
        ord_names = ordinal_cols
        
        feature_names = np.r_[num_names, nom_names, ord_names]
        
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    except Exception as e:
        # print(f"Advertencia: No se pudieron recuperar los nombres de las features: {e}")
        X_train_df = pd.DataFrame(X_train_processed)
        X_test_df = pd.DataFrame(X_test_processed)

    # print("Ingeniería de características completada.")
    # print(f"Shape Train: {X_train_df.shape}")
    # print(f"Shape Test: {X_test_df.shape}")
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Bloque de prueba
    try:
        # Ajustar ruta relativa subiendo niveles si es necesario, o asumiendo ejecución desde root
        ruta_db = Path("Base_de_datos.xlsx")
        # Si se ejecuta desde src/
        if not ruta_db.exists():
             ruta_db = Path("../../Base_de_datos.xlsx")
             
        print(f"Cargando datos desde: {ruta_db.resolve()}")
        df = cargar_datos(ruta_db)
        
        X_train, X_test, y_train, y_test, prep = feature_engineering(df)
        
        print("\nEjemplo de X_train procesado:")
        print(X_train.head())
        
    except Exception as e:
        print(f"Error en ejecución principal: {e}")
