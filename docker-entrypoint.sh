#!/bin/bash

echo "🚀 Iniciando aplicación ML Dashboard..."

# Esperar a que la base de datos esté lista (si usas PostgreSQL)
echo "⏳ Esperando base de datos..."
sleep 3

# Ejecutar migraciones
echo "📦 Aplicando migraciones..."
python manage.py makemigrations --noinput
python manage.py migrate --noinput

# Crear superusuario si no existe
echo "👤 Verificando superusuario..."
python manage.py shell << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('✅ Superusuario creado: admin/admin123')
else:
    print('ℹ️  Superusuario ya existe')
END

# Crear código de acceso por defecto
echo "🔑 Verificando código de acceso..."
python manage.py shell << END
from apps.accounts.models import CodigoAcceso
from datetime import datetime, timedelta
if not CodigoAcceso.objects.filter(codigo='ESTUDIANTE2024').exists():
    CodigoAcceso.objects.create(
        codigo='ESTUDIANTE2024',
        descripcion='Código para estudiantes',
        usos_maximos=100,
        activo=True
    )
    print('✅ Código de acceso creado: ESTUDIANTE2024')
else:
    print('ℹ️  Código de acceso ya existe')
END

# Recolectar archivos estáticos
echo "📂 Recolectando archivos estáticos..."
python manage.py collectstatic --noinput

echo "✅ Inicialización completada!"
echo "🌐 Iniciando servidor Gunicorn..."

# Iniciar Gunicorn
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
