# Usar Python 3.11 como base
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar gunicorn para producción
RUN pip install gunicorn whitenoise

# Copiar proyecto
COPY . .

# Crear directorios necesarios
RUN mkdir -p staticfiles media/avatars media/datasets media/modelos

# Recolectar archivos estáticos
RUN python manage.py collectstatic --noinput || true

# Exponer puerto
EXPOSE 8000

# Script de inicio
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Comando de inicio
ENTRYPOINT ["/app/docker-entrypoint.sh"]
