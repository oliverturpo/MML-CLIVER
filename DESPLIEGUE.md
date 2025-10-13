# 🚀 Guía de Despliegue - ML Dashboard en Hostinger con Easypanel

## 📋 Pre-requisitos

- ✅ VPS en Hostinger
- ✅ Easypanel instalado
- ✅ Dominio configurado (opcional)
- ✅ Git instalado

---

## 🎯 Opción 1: Despliegue con Easypanel (Recomendado)

### **Paso 1: Subir código a GitHub**

```bash
# En tu máquina local
cd /mnt/f/APRENDIZAJE/SUPERVISADO-DASHBOARD

# Inicializar git (si no lo has hecho)
git init
git add .
git commit -m "Initial commit - ML Dashboard ready for deployment"

# Crear repositorio en GitHub y subir
git remote add origin https://github.com/TU-USUARIO/ml-dashboard.git
git branch -M main
git push -u origin main
```

### **Paso 2: Configurar en Easypanel**

1. **Acceder a Easypanel**
   - URL: `https://tu-vps-ip:3000` o tu dominio configurado
   - Login con tus credenciales

2. **Crear Nueva Aplicación**
   - Click en "Create App"
   - Nombre: `ml-dashboard`
   - Tipo: `Docker`

3. **Configurar Git**
   - Repository URL: `https://github.com/TU-USUARIO/ml-dashboard.git`
   - Branch: `main`
   - Build Method: `Dockerfile`

4. **Variables de Entorno** (Environment Variables)
   ```
   DEBUG=False
   DJANGO_SECRET_KEY=tu-clave-super-secreta-aqui-cambiarla
   ALLOWED_HOSTS=tu-dominio.com,www.tu-dominio.com,tu-vps-ip
   ```

5. **Configurar Puerto**
   - Internal Port: `8000`
   - External Port: `80` (o `443` si tienes SSL)

6. **Volumes (Persistencia de Datos)**
   ```
   /app/media -> /data/ml-dashboard/media
   /app/db.sqlite3 -> /data/ml-dashboard/db.sqlite3
   ```

7. **Deploy**
   - Click en "Deploy"
   - Esperar 2-5 minutos

### **Paso 3: Verificar Despliegue**

1. Acceder a: `http://tu-dominio.com` o `http://tu-vps-ip`
2. Login admin: `admin` / `admin123`
3. Código estudiantes: `ESTUDIANTE2024`

---

## 🐳 Opción 2: Despliegue Manual con Docker

### **En tu VPS (SSH):**

```bash
# 1. Conectar a tu VPS
ssh root@tu-vps-ip

# 2. Instalar Docker (si no está instalado)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Instalar Docker Compose
apt-get install docker-compose -y

# 4. Clonar tu repositorio
git clone https://github.com/TU-USUARIO/ml-dashboard.git
cd ml-dashboard

# 5. Configurar variables de entorno
nano .env
```

**Contenido de .env:**
```env
DEBUG=False
DJANGO_SECRET_KEY=cambia-esto-por-una-clave-segura
ALLOWED_HOSTS=tu-dominio.com,tu-vps-ip
```

```bash
# 6. Construir y ejecutar
docker-compose up -d --build

# 7. Ver logs
docker-compose logs -f
```

### **Acceder a la aplicación:**
- URL: `http://tu-vps-ip:8000`

---

## 🔧 Configuración de Nginx (Opcional - Recomendado)

Para usar dominio y SSL:

```bash
# 1. Instalar Nginx
apt-get install nginx -y

# 2. Configurar Nginx
nano /etc/nginx/sites-available/ml-dashboard
```

**Contenido:**
```nginx
server {
    listen 80;
    server_name tu-dominio.com www.tu-dominio.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /ruta/a/ml-dashboard/staticfiles/;
    }

    location /media/ {
        alias /ruta/a/ml-dashboard/media/;
    }
}
```

```bash
# 3. Activar configuración
ln -s /etc/nginx/sites-available/ml-dashboard /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# 4. Instalar SSL (Let's Encrypt)
apt-get install certbot python3-certbot-nginx -y
certbot --nginx -d tu-dominio.com -d www.tu-dominio.com
```

---

## 🔐 Configuración de Seguridad Post-Despliegue

### **1. Cambiar credenciales por defecto**

```bash
# Acceder al contenedor
docker exec -it ml-dashboard_web_1 bash

# Cambiar password admin
python manage.py changepassword admin

# Crear nuevo código de acceso desde admin panel
```

### **2. Generar SECRET_KEY segura**

```python
# En Python local
import secrets
print(secrets.token_urlsafe(50))
```

Actualiza la variable `DJANGO_SECRET_KEY` en Easypanel o `.env`

### **3. Configurar Firewall**

```bash
# UFW en Ubuntu/Debian
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

---

## 📊 Monitoreo y Mantenimiento

### **Ver logs en tiempo real:**

**Con Easypanel:**
- Dashboard → ml-dashboard → Logs

**Con Docker:**
```bash
docker-compose logs -f web
```

### **Backup de Base de Datos:**

```bash
# Dentro del contenedor
docker exec ml-dashboard_web_1 python manage.py dumpdata > backup.json

# Copiar a local
docker cp ml-dashboard_web_1:/app/backup.json ./backup.json
```

### **Actualizar aplicación:**

**Con Easypanel:**
1. Push cambios a GitHub
2. En Easypanel: Click "Rebuild"

**Con Docker:**
```bash
git pull
docker-compose down
docker-compose up -d --build
```

---

## 🌐 URLs Importantes Post-Despliegue

- **Landing Page**: `https://tu-dominio.com/`
- **Admin Panel**: `https://tu-dominio.com/admin/`
- **Registro**: `https://tu-dominio.com/accounts/registro/`

### **Credenciales por Defecto:**
- **Admin**: `admin` / `admin123` (⚠️ CAMBIAR INMEDIATAMENTE)
- **Código estudiantes**: `ESTUDIANTE2024`

---

## 🐛 Solución de Problemas

### **Error: "Bad Gateway" o 502**
```bash
# Verificar que el contenedor esté corriendo
docker ps

# Ver logs
docker-compose logs web

# Reiniciar
docker-compose restart
```

### **Archivos estáticos no cargan**
```bash
# Dentro del contenedor
docker exec ml-dashboard_web_1 python manage.py collectstatic --noinput
```

### **Permisos de media/**
```bash
chmod -R 755 media/
chown -R www-data:www-data media/
```

---

## 📝 Checklist Final

- [ ] Código subido a GitHub
- [ ] App desplegada en Easypanel/Docker
- [ ] Variables de entorno configuradas
- [ ] SECRET_KEY cambiada
- [ ] Password admin cambiado
- [ ] Dominio configurado (opcional)
- [ ] SSL instalado (opcional)
- [ ] Backup automático configurado
- [ ] Firewall activado
- [ ] Logs monitoreados

---

## 🎓 Para Estudiantes

**Código de Acceso**: `ESTUDIANTE2024`

### **Cómo Registrarse:**
1. Ir a: `https://tu-dominio.com/accounts/registro/`
2. Ingresar código: `ESTUDIANTE2024`
3. Crear PIN de 4 dígitos
4. Listo!

### **Cómo Acceder:**
1. Landing Page → Buscar tu perfil
2. Click "Ver Proyecto"
3. Ingresar tu PIN

---

## 📧 Soporte

Si tienes problemas:
1. Revisa los logs
2. Verifica las variables de entorno
3. Consulta la documentación oficial de Django/Docker

**Hecho por Estudiantes de la FINESI** 🎓
