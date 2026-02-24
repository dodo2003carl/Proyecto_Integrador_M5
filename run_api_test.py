import subprocess
import time
import requests
import json
import os
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "."
print("Iniciando servidor de la API (FastAPI + Uvicorn)...")
process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8080"],
    env=env
)

time.sleep(4) # Esperar a que el servidor inicialice los modelos

payload = [
    {
        "tipo_credito": 1.0,
        "capital_prestado": 15000.0,
        "plazo_meses": 36.0,
        "edad_cliente": 28.0,
        "salario_cliente": 4500.0,
        "total_otros_prestamos": 0.0,
        "cuota_pactada": 500.0,
        "puntaje": 80.0,
        "puntaje_datacredito": 850.0,
        "cant_creditosvigentes": 1.0,
        "huella_consulta": 1.0,
        "saldo_mora": 0.0,
        "saldo_total": 5000.0,
        "saldo_principal": 5000.0,
        "saldo_mora_codeudor": 0.0,
        "creditos_sectorFinanciero": 1.0,
        "creditos_sectorCooperativo": 0.0,
        "creditos_sectorReal": 0.0,
        "promedio_ingresos_datacredito": 4500.0,
        "tipo_laboral": "Fijo",
        "tendencia_ingresos": "Creciente"
    }
]

print("\n---------------------------------------------------------")
print("Enviando cliente simulado (POST) a http://localhost:8080/predict...")
print(json.dumps(payload[0], indent=2, ensure_ascii=False))
print("---------------------------------------------------------")

try:
    response = requests.post("http://localhost:8080/predict", json=payload)
    print("\n✅ Status Code:", response.status_code)
    print("✅ Respuesta de la API (Predicción del Modelo):")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
except Exception as e:
    print("❌ Error conectando a la API:", e)
finally:
    print("\nCerrando servidor de la API...")
    process.terminate()
    process.wait()
    print("Prueba completada.")
