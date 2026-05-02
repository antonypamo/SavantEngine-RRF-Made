# ============================
# Dockerfile – Savant RRF Φ12.0 API
# ============================
FROM python:3.11-slim

# Evitar pyc y usar stdout sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema (para numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalarlos
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API
COPY app.py /app/app.py

# Puerto por defecto de Hugging Face Spaces para FastAPI
EXPOSE 7860

# Comando de arranque
# Nota: host 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
