from fastapi.testclient import TestClient
from mlops_pipeline.src.model_deploy import app

client = TestClient(app)

def test_home():
    """Prueba que el endpoint raíz responda correctamente"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenido" in response.json()["message"]

def test_predict_endpoint_success():
    """Prueba una predicción exitosa enviando datos estructurados de cliente"""
    # Payload simulado de un cliente
    payload = [
        {
            "tipo_credito": 1.0,
            "capital_prestado": 5000.0,
            "plazo_meses": 24.0,
            "edad_cliente": 30.0,
            "salario_cliente": 2500.0,
            "total_otros_prestamos": 1000.0,
            "cuota_pactada": 250.0,
            "puntaje": 75.0,
            "puntaje_datacredito": 750.0,
            "cant_creditosvigentes": 1.0,
            "huella_consulta": 0.0,
            "saldo_mora": 0.0,
            "saldo_total": 6000.0,
            "saldo_principal": 5000.0,
            "saldo_mora_codeudor": 0.0,
            "creditos_sectorFinanciero": 1.0,
            "creditos_sectorCooperativo": 0.0,
            "creditos_sectorReal": 0.0,
            "promedio_ingresos_datacredito": 2500.0,
            "tipo_laboral": "Fijo",
            "tendencia_ingresos": "Estable"
        }
    ]
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    
    # Verificar que los campos esperados estén en la respuesta
    resultado = data[0]
    assert "prediccion" in resultado
    assert "probabilidad_pago" in resultado
    assert "interpretacion" in resultado
    
    # Solo puede ser 1 o 0
    assert resultado["prediccion"] in [0, 1]

def test_predict_endpoint_validation_error():
    """Prueba que el endpoint falle correctamente si falta un campo (Pydantic ValidationError)"""
    payload_incompleto = [
        {
            "tipo_credito": 1.0,
            "capital_prestado": 5000.0
            # Faltan todos los demás campos obligatorios
        }
    ]
    
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422 # 422 Unprocessable Entity de FastAPI
