import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos la función de ingeniería de características
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ft_engineering import cargar_datos, feature_engineering

def evaluate_model(model_name, model, X_test, y_test):
    """Evalúa un modelo y retorna sus métricas clave."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, classification_report(y_test, y_pred, zero_division=0)

def main():
    print("==================================================")
    print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN")
    print("==================================================\n")
    
    # 1. Cargar Datos
    print("[1/4] Cargando datos y ejecutando Ingeniería de Características...")
    try:
        ruta_db = Path("Base_de_datos.xlsx")
        if not ruta_db.exists():
            ruta_db = Path("../../Base_de_datos.xlsx")
        
        df = cargar_datos(ruta_db)
        X_train, X_test, y_train, y_test, preprocessor = feature_engineering(df)
        print(f"      ✅ Datos cargados. Shape Train: {X_train.shape}, Shape Test: {X_test.shape}\n")
    except Exception as e:
        print(f"      ❌ Error al cargar o procesar datos: {e}")
        return

    # 2. Entrenar Múltiples Modelos
    print("[2/4] Entrenando múltiples modelos experimentales...")
    models = {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    }
    
    trained_models = {}
    metrics_list = []
    reports = {}
    
    for name, model in models.items():
        print(f"      > Entrenando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluar
        metrics, report = evaluate_model(name, model, X_test, y_test)
        metrics_list.append(metrics)
        reports[name] = report

    # 3. Comparación y Evaluación
    print("\n[3/4] Evaluando Modelos - Tabla de Comparación Crítica:")
    metrics_df = pd.DataFrame(metrics_list)
    print("\n" + metrics_df.to_string(index=False))
    
    # Elegir el mejor modelo basado en F1-Score (balance entre Precision y Recall)
    best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax()]['Modelo']
    best_model = trained_models[best_model_name]
    
    print("\n==================================================")
    print(f"🏆 MEJOR MODELO SELECCIONADO: **{best_model_name.upper()}**")
    print(f"🏆 Justificación: Presenta el mejor balance (F1-Score) para clasificar riesgo.")
    print("==================================================\n")
    
    print(f"Reporte de Clasificación Detallado ({best_model_name}):\n")
    print(reports[best_model_name])
    
    # 4. Guardar el Mejor Modelo
    print("\n[4/4] Serializando y Guardando el Mejor Modelo para Producción...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True, parents=True) # Crea local models
    
    joblib.dump(best_model, model_dir / "modelo_final.pkl")
    joblib.dump(preprocessor, model_dir / "preprocesador.pkl")
    print(f"      ✅ Mejor modelo guardado en: {model_dir / 'modelo_final.pkl'}")
    
    print("\n🚀 PIPELINE COMPLETADO EXITOSAMENTE.")

if __name__ == "__main__":
    main()
