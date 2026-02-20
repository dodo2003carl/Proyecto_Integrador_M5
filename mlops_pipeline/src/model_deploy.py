from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import joblib
from pathlib import Path

# Configuración de FastAPI
app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="API que utiliza un modelo de Machine Learning para predecir si un cliente pagará a tiempo.",
    version="1.0.0"
)

# Cargar los modelos al iniciar la aplicación
MODEL_DIR = Path("models")
if not MODEL_DIR.exists():
    # If run from src/
    MODEL_DIR = Path("../../models")

model_path = MODEL_DIR / "modelo_final.pkl"
preprocessor_path = MODEL_DIR / "preprocesador.pkl"

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Modelo y preprocesador cargados correctamente.")
except Exception as e:
    model = None
    preprocessor = None
    print(f"Error al cargar modelos: {e}. Asegúrese de ejecutar save_model.py primero.")

# Esquema de datos de entrada usando Pydantic
class ClientData(BaseModel):
    tipo_credito: float = Field(..., description="Tipo de crédito numérico")
    capital_prestado: float = Field(..., description="Capital prestado")
    plazo_meses: float = Field(..., description="Plazo del préstamo en meses")
    edad_cliente: float = Field(..., description="Edad del cliente")
    salario_cliente: float = Field(..., description="Salario del cliente")
    total_otros_prestamos: float = Field(..., description="Total en otros préstamos")
    cuota_pactada: float = Field(..., description="Cuota mensual pactada")
    puntaje: float = Field(..., description="Puntaje crediticio interno")
    puntaje_datacredito: float = Field(..., description="Puntaje de bureau (DataCrédito)")
    cant_creditosvigentes: float = Field(..., description="Cantidad de créditos vigentes")
    huella_consulta: float = Field(..., description="Número de consultas recientes")
    saldo_mora: float = Field(..., description="Saldo actual en mora")
    saldo_total: float = Field(..., description="Saldo total de endeudamiento")
    saldo_principal: float = Field(..., description="Saldo principal")
    saldo_mora_codeudor: float = Field(..., description="Mora como codeudor")
    creditos_sectorFinanciero: float = Field(..., description="Créditos en sector financiero")
    creditos_sectorCooperativo: float = Field(..., description="Créditos en sector cooperativo")
    creditos_sectorReal: float = Field(..., description="Créditos en sector real")
    promedio_ingresos_datacredito: float = Field(..., description="Promedio de ingresos reportados")
    tipo_laboral: str = Field(..., description="Tipo de ocupación o contrato")
    tendencia_ingresos: str = Field(..., description="Tendencia de ingresos (ej. Crece, Baja, Estable)")

class PredictionResult(BaseModel):
    prediccion: int
    probabilidad_pago: float
    interpretacion: str

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Riesgo Crediticio. Visite /docs para la documentación."}

@app.post("/predict", response_model=List[PredictionResult])
def predict(clientes: List[ClientData]):
    """
    Recibe una lista de clientes (Batch) y devuelve las predicciones.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Los modelos no están cargados en el servidor.")
        
    try:
        # Convertir datos de entrada a DataFrame (Batch)
        df_input = pd.DataFrame([vars(cliente) for cliente in clientes])
        
        # Aplicar el preprocesador (Transformar texto a números/dummies, etc.)
        X_processed = preprocessor.transform(df_input)
        
        # Realizar predicción
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1] # Probabilidad de la clase 1 (Pago_atiempo=1)
        
        resultados = []
        for pred, prob in zip(predictions, probabilities):
            resultados.append({
                "prediccion": int(pred),
                "probabilidad_pago": float(prob),
                "interpretacion": "Pagador a Tiempo" if pred == 1 else "Riesgo de Impago"
            })
            
        return resultados
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")

# Si se ejecuta directamente (para pruebas locales)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)
