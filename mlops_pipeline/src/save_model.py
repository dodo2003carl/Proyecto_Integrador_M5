import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from ft_engineering import cargar_datos, feature_engineering
import platform

def train_and_save():
    print("Iniciando carga de datos...")
    
    # Try different paths depending on where it's run from
    ruta_db = Path("Base_de_datos.xlsx")
    if not ruta_db.exists():
        ruta_db = Path("../../Base_de_datos.xlsx")

    df = cargar_datos(ruta_db)
    
    print("Ejecutando ingeniería de características...")
    X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df)
    
    print("Entrenando el modelo (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Precisión en Test: {model.score(X_test, y_test):.4f}")
    
    print("Guardando el modelo y preprocesador...")
    # Creamos un directorio de modelos
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, model_dir / "modelo_final.pkl")
    joblib.dump(preprocessor, model_dir / "preprocesador.pkl")
    
    print("¡Modelos guardados con éxito en la carpeta 'models'!")

if __name__ == "__main__":
    train_and_save()
