# Usar una imagen base oficial de Python ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto al contenedor (incluyendo modelos y src)
COPY mlops_pipeline/ ./mlops_pipeline/
COPY models/ ./models/

# Exponer el puerto en el que correrá FastAPI
EXPOSE 8000

# Comando para iniciar el servidor Uvicorn apuntando a app en model_deploy.py
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
