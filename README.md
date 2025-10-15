# Dashboard de Aprendizaje Supervisado - Machine Learning

Sistema web educativo para el entrenamiento, visualización y análisis de modelos de Machine Learning con enfoque en Aprendizaje Supervisado. Desarrollado con Django 5.2.7 y scikit-learn.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Tecnologías Utilizadas](#tecnologías-utilizadas)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Perfiles Implementados](#perfiles-implementados)
5. [Algoritmos Soportados](#algoritmos-soportados)
6. [Métricas y Evaluación](#métricas-y-evaluación)
7. [Instalación](#instalación)
8. [Uso del Sistema](#uso-del-sistema)
9. [Arquitectura](#arquitectura)
10. [Base de Datos](#base-de-datos)
11. [Funcionalidades Principales](#funcionalidades-principales)
12. [Despliegue](#despliegue)

---

## Descripción General

**Dashboard de Aprendizaje Supervisado** es una plataforma web interactiva diseñada para estudiantes y profesionales que desean aprender y practicar algoritmos de Machine Learning de forma visual e intuitiva.

### Características Principales

- **Sistema de Perfiles Personalizados**: Cada estudiante tiene un perfil único con su proyecto específico
- **Autenticación por PIN**: Sistema de acceso simplificado con código PIN de 4 dígitos
- **Carga de Datasets**: Soporte para archivos CSV con preprocesamiento automático
- **Entrenamiento de Modelos**: 10+ algoritmos de clasificación y regresión
- **Visualización de Resultados**: Métricas, gráficos interactivos y matrices de confusión
- **Validación Cruzada**: Evaluación robusta con k-fold cross-validation
- **Plantillas Personalizadas**: Resultados adaptados según el tipo de proyecto y algoritmo

---

## Tecnologías Utilizadas

### Backend
- **Django 5.2.7** - Framework web principal
- **Python 3.x** - Lenguaje de programación
- **SQLite** - Base de datos (desarrollo)
- **PostgreSQL** - Base de datos (producción)

### Machine Learning
- **scikit-learn** - Biblioteca principal de ML
  - LinearRegression
  - LogisticRegression
  - Ridge / Lasso
  - DecisionTreeClassifier / Regressor
  - RandomForestClassifier / Regressor
  - SVC / SVR
  - KNeighborsClassifier / Regressor
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **matplotlib** - Generación de gráficos
- **seaborn** - Visualizaciones estadísticas

### Frontend
- **Tailwind CSS 3.x** - Framework de estilos
- **Font Awesome 6** - Iconografía
- **Chart.js** - Gráficos interactivos (opcional)
- **HTML5 + JavaScript** - Interfaz de usuario



## Estructura del Proyecto

```
SUPERVISADO-DASHBOARD/
├── apps/
│   ├── accounts/              # Autenticación y gestión de usuarios
│   │   ├── models.py          # CodigoAcceso, Perfil
│   │   ├── views.py           # Registro, login con PIN
│   │   ├── urls.py
│   │   └── admin.py
│   │
│   ├── dashboard/             # Dashboard principal
│   │   ├── models.py          # (Modelos compartidos)
│   │   ├── views.py           # Inicio, datos, modelo, resultados
│   │   ├── urls.py
│   │   └── admin.py
│   │
│   └── ml_models/             # Motor de Machine Learning
│       ├── models.py          # (Sin modelos de BD)
│       ├── views.py           # Pipeline de entrenamiento ML
│       └── utils.py           # Funciones auxiliares
│
├── config/                    # Configuración del proyecto
│   ├── settings.py            # Configuración de Django
│   ├── urls.py                # URLs principales
│   ├── wsgi.py
│   └── asgi.py
│
├── templates/
│   ├── base.html              # Plantilla base
│   ├── accounts/
│   │   ├── login_pin.html     # Login con PIN
│   │   └── registro.html      # Registro de estudiantes
│   │
│   └── dashboard/
│       ├── base_dashboard.html      # Base del dashboard
│       ├── landing.html             # Landing page pública
│       ├── inicio.html              # Dashboard principal
│       ├── config.html              # Configuración del perfil
│       ├── datos.html               # Carga de datasets
│       ├── modelo.html              # Selección y entrenamiento
│       ├── resultados.html          # Resultados generales
│       ├── resultados_perfil1.html  # Cliver - Pobreza Puno
│       ├── resultados_perfil4.html  # Noemi - Rendimiento Académico
│       └── resultados_perfil5.html  # Zulema - Anemia Gestantes
│
├── static/                    # Archivos estáticos
├── media/                     # Archivos subidos por usuarios
│   ├── datasets/              # CSVs subidos
│   └── avatars/               # Fotos de perfil
│
├── db.sqlite3                 # Base de datos SQLite (desarrollo)
├── manage.py                  # CLI de Django
├── requirements.txt           # Dependencias Python

└── README.md                  # Este archivo
```

---

## Perfiles Implementados

El sistema soporta perfiles personalizados para cada estudiante con proyectos específicos:

### Perfil 1 - Cliver (Pobreza en Puno)
- **Proyecto**: Predicción de Niveles de Pobreza en Puno 2025
- **Algoritmo**: Regresión Logística Multiclase
- **Dataset**: `dataset_pobreza_puno.csv`
- **Variables**:
  - Edad
  - Ingresos Mensuales
  - Nivel Educativo
  - Tamaño Hogar
  - Acceso a Servicios Básicos
  - Distrito
- **Target**: Nivel de Pobreza (Extrema, Moderada, No Pobre)
- **Plantilla**: `resultados_perfil1.html`

### Perfil 4 - Noemi (Rendimiento Académico)
- **Proyecto**: Predicción de Rendimiento Académico Estudiantil
- **Algoritmo**: Regresión Lineal Múltiple
- **Dataset**: Datos de estudiantes universitarios
- **Variables**:
  - Puntaje Inicial
  - Asistencia a Clases
  - Horas de Estudio Semanal
  - Uso de Redes Sociales (distractor)
- **Target**: Calificación Final (continua)
- **Métricas**: R² Score, RMSE, MAE, MSE
- **Plantilla**: `resultados_perfil4.html`

### Perfil 5 - Zulema (Anemia en Gestantes)
- **Proyecto**: Clasificación de Anemia en Gestantes - Puno 2025
- **Algoritmo**: Regresión Logística Multiclase
- **Dataset**: Datos de gestantes del sistema de salud
- **Variables**:
  - Edad de la gestante (15-45 años)
  - Edad gestacional (semanas de embarazo)
  - Provincia de residencia
  - Hemoglobina (g/dL)
- **Target**: Diagnóstico de Anemia (Normal, Leve, Moderada, Severa)
- **Criterios OMS**:
  - Normal: ≥ 11.0 g/dL
  - Anemia Leve: 10.0 - 10.9 g/dL
  - Anemia Moderada: 7.0 - 9.9 g/dL
  - Anemia Severa: < 7.0 g/dL
- **Métricas**: Accuracy, Precision, Recall, F1-Score, Matriz de Confusión 4x4
- **Plantilla**: `resultados_perfil5.html`

---

## Algoritmos Soportados

### Algoritmos de Regresión
1. **Regresión Lineal** (`regresion_lineal`)
   - Modelo lineal clásico
   - Coeficientes interpretables
   - Métricas: R², RMSE, MAE, MSE

2. **Regresión Polinomial** (`regresion_polinomial`)
   - Relaciones no lineales
   - Transformación polinomial de características

3. **Ridge Regression** (`ridge`)
   - Regularización L2
   - Previene overfitting
   - Coeficientes penalizados

4. **Lasso Regression** (`lasso`)
   - Regularización L1
   - Selección automática de características

### Algoritmos de Clasificación
1. **Regresión Logística** (`regresion_logistica`)
   - Clasificación binaria y multiclase
   - Probabilidades interpretables
   - Matriz de confusión

2. **Árbol de Decisión** (`decision_tree`)
   - Clasificación jerárquica
   - Importancia de características
   - Fácil interpretación

3. **Random Forest** (`random_forest`)
   - Ensemble de árboles
   - Alta precisión
   - Robustez contra overfitting

4. **Support Vector Machine** (`svm`)
   - Clasificación con márgenes
   - Kernel RBF para no linealidad

5. **K-Nearest Neighbors** (`knn`)
   - Clasificación basada en vecinos
   - No paramétrico
   - Ajustable (k=5 por defecto)

---

## Métricas y Evaluación

### Métricas de Regresión
- **R² Score (Coeficiente de Determinación)**
  - Rango: -∞ a 1.0
  - Interpretación: % de varianza explicada
  - Valores recomendados: > 0.7 (Bueno), > 0.5 (Aceptable)

- **RMSE (Root Mean Squared Error)**
  - Error típico de predicción
  - Misma unidad que el target
  - Menor es mejor

- **MAE (Mean Absolute Error)**
  - Error absoluto promedio
  - Menos sensible a outliers que RMSE

- **MSE (Mean Squared Error)**
  - Error cuadrático medio
  - Penaliza errores grandes

### Métricas de Clasificación
- **Accuracy (Exactitud)**
  - % de predicciones correctas
  - Rango: 0 a 1.0

- **Precision (Precisión)**
  - % de positivos predichos que son correctos
  - Importante cuando los falsos positivos son costosos

- **Recall (Sensibilidad)**
  - % de positivos reales que se detectan
  - Importante cuando los falsos negativos son costosos

- **F1-Score**
  - Media armónica de Precision y Recall
  - Balance entre ambas métricas

- **Matriz de Confusión**
  - Visualización de aciertos y errores
  - Soporta clasificación multiclase (hasta 4x4)

### Validación Cruzada
- **5-Fold Cross Validation**
  - División en 5 partes
  - Entrenamiento iterativo
  - Métricas promedio y desviación estándar
  - Evaluación de generalización

---

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git
- (Opcional) Docker y Docker Compose

### Instalación Local

#### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd SUPERVISADO-DASHBOARD
```

#### 2. Crear entorno virtual
```bash
python -m venv venv
```

#### 3. Activar entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

#### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 5. Configurar variables de entorno
Crear archivo `.env` en la raíz del proyecto:
```env
SECRET_KEY=tu-clave-secreta-aqui
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:///db.sqlite3
```

#### 6. Realizar migraciones
```bash
python manage.py makemigrations
python manage.py migrate
```

#### 7. Crear superusuario (administrador)
```bash
python manage.py createsuperuser
```

#### 8. Crear código de acceso inicial
Acceder al admin en `http://localhost:8000/admin/` y crear un CodigoAcceso con:
- **Código**: `CICLO7-2025`
- **Límite de usos**: 50
- **Activo**: Sí

#### 9. Ejecutar el servidor
```bash
python manage.py runserver
```

#### 10. Acceder al sistema
- **Landing Page**: http://localhost:8000/
- **Admin Panel**: http://localhost:8000/admin/
- **Registro**: http://localhost:8000/registro/

---

## Uso del Sistema

### Para Estudiantes

#### 1. Registro
1. Ir a http://localhost:8000/registro/
2. Completar el formulario:
   - Nombre completo
   - Email universitario
   - Carrera
   - Código del estudiante
   - Código de acceso (proporcionado por el docente)
   - PIN de 4 dígitos (para login)
3. Hacer clic en "Crear Perfil"

#### 2. Login con PIN
1. En la landing page (http://localhost:8000/), buscar tu perfil
2. Hacer clic en "Ver Proyecto"
3. Ingresar el PIN de 4 dígitos
4. Acceso al dashboard personalizado

#### 3. Configuración del Perfil
1. Ir a la sección "Configuración"
2. Configurar:
   - Título del proyecto
   - Descripción
   - Foto de perfil (opcional)
   - Información académica

#### 4. Carga de Dataset
1. Ir a la sección "Datos"
2. Hacer clic en "Subir Dataset"
3. Seleccionar archivo CSV
   - **Importante**: La última columna debe ser la variable objetivo (target)
   - Formato: UTF-8 o Latin-1
   - Tamaño máximo: 10 MB
4. Ver preview y estadísticas del dataset

#### 5. Selección de Algoritmo
1. Ir a la sección "Modelo"
2. Seleccionar algoritmo de la lista:
   - Regresión: Linear, Polinomial, Ridge, Lasso
   - Clasificación: Logística, Árbol, Random Forest, SVM, KNN
3. Configurar parámetros:
   - Tamaño del conjunto de prueba (20-30% recomendado)

#### 6. Entrenamiento del Modelo
1. Hacer clic en "Entrenar Modelo"
2. El sistema automáticamente:
   - Preprocesa los datos (manejo de nulos, encoding categórico)
   - Divide en train/test
   - Entrena el modelo
   - Calcula métricas de evaluación
   - Realiza validación cruzada
   - Genera visualizaciones
3. Ver tiempo de entrenamiento

#### 7. Visualización de Resultados
1. Ir a la sección "Resultados"
2. Ver métricas principales:
   - Para regresión: R², RMSE, MAE, MSE, coeficientes
   - Para clasificación: Accuracy, Precision, Recall, F1-Score, matriz de confusión
3. Analizar gráficos:
   - Importancia de características
   - Distribución de errores
   - Validación cruzada
4. Interpretación contextualizada según el proyecto

### Para Docentes

#### 1. Acceso al Panel de Administración
1. Ir a http://localhost:8000/admin/
2. Login con credenciales de superusuario
3. Acceso a:
   - Códigos de Acceso
   - Perfiles de Estudiantes
   - Configuración del sistema

#### 2. Gestión de Códigos de Acceso
1. Ir a "Códigos de Acceso"
2. Crear nuevo código:
   - Código único (ej: "CICLO7-2025")
   - Descripción (ej: "Curso de Machine Learning - Ciclo VII")
   - Límite de usos (cantidad de estudiantes permitidos)
   - Fecha de expiración (opcional)
   - Activo (Sí/No)
3. Compartir el código con los estudiantes
4. Monitorear usos restantes

#### 3. Supervisión de Estudiantes
1. Ir a "Perfiles"
2. Ver lista de estudiantes registrados
3. Hacer clic en un perfil para ver:
   - Información personal
   - Dataset cargado
   - Algoritmo seleccionado
   - Métricas del modelo
   - Tiempo de entrenamiento
   - Fecha de última actividad
4. Editar o eliminar perfiles si es necesario

---

## Arquitectura

### Diagrama de Flujo del Sistema

```
┌─────────────────┐
│   Usuario       │
│  (Estudiante)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Landing Page                    │
│  - Ver perfiles públicos            │
│  - Buscar proyecto                  │
│  - Login con PIN                    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Dashboard (views.py)            │
│  - dashboard_inicio                 │
│  - dashboard_config                 │
│  - dashboard_datos                  │
│  - dashboard_modelo                 │
│  - dashboard_resultados             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ML Pipeline (ml_models/views.py)  │
│                                     │
│  1. Preprocesamiento                │
│     - LabelEncoder (categóricas)    │
│     - StandardScaler                │
│     - Manejo de valores nulos       │
│                                     │
│  2. División de datos               │
│     - train_test_split              │
│                                     │
│  3. Selección de algoritmo          │
│     - Regresión / Clasificación     │
│                                     │
│  4. Entrenamiento                   │
│     - model.fit()                   │
│                                     │
│  5. Predicción                      │
│     - model.predict()               │
│                                     │
│  6. Evaluación                      │
│     - Métricas según tipo           │
│     - Validación cruzada (5-fold)   │
│                                     │
│  7. Almacenamiento                  │
│     - Guardar en perfil.metricas    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Resultados (templates/)           │
│  - resultados.html (genérico)       │
│  - resultados_perfil1.html          │
│  - resultados_perfil4.html          │
│  - resultados_perfil5.html          │
└─────────────────────────────────────┘
```

### Modelo de Datos

#### Modelo `CodigoAcceso`
```python
class CodigoAcceso(models.Model):
    codigo = models.CharField(max_length=50, unique=True)
    descripcion = models.TextField()
    limite_usos = models.IntegerField(default=1)
    usos_actuales = models.IntegerField(default=0)
    activo = models.BooleanField(default=True)
    fecha_expiracion = models.DateTimeField(null=True, blank=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
```

#### Modelo `Perfil`
```python
class Perfil(models.Model):
    # Información Personal
    nombre_completo = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    codigo_estudiante = models.CharField(max_length=50)
    carrera = models.CharField(max_length=200)

    # Autenticación
    pin = models.CharField(max_length=4)

    # Proyecto
    titulo_proyecto = models.CharField(max_length=300)
    descripcion_proyecto = models.TextField()
    foto_perfil = models.ImageField(upload_to='avatars/')

    # Dataset
    dataset_nombre = models.CharField(max_length=255)
    dataset_archivo = models.FileField(upload_to='datasets/')

    # Modelo ML
    algoritmo_elegido = models.CharField(max_length=50)
    tamaño_test = models.FloatField(default=0.3)
    tiempo_entrenamiento = models.CharField(max_length=50)

    # Métricas (JSONField)
    metricas = models.JSONField(default=dict, blank=True)
    # Ejemplo de estructura:
    # {
    #   "Accuracy": 0.9673,
    #   "Precision": 0.9681,
    #   "Recall": 0.9673,
    #   "F1_Score": 0.9673,
    #   "Confusion_Matrix": [[...], [...], ...],
    #   "CV_Mean_Score": 0.9383,
    #   "CV_Std_Score": 0.0145,
    #   "Train_Size": 400,
    #   "Test_Size": 100
    # }

    # Fechas
    fecha_registro = models.DateTimeField(auto_now_add=True)
    ultima_actividad = models.DateTimeField(auto_now=True)

    # Relaciones
    codigo_acceso = models.ForeignKey(CodigoAcceso)
```

---

## Base de Datos

### Desarrollo
- **SQLite** (`db.sqlite3`)
- Sin configuración adicional
- Ideal para desarrollo local

### Producción
- **PostgreSQL** (recomendado)
- Configuración en `config/settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'supervisado_db',
        'USER': 'postgres',
        'PASSWORD': 'tu_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Migraciones

```bash
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Ver migraciones aplicadas
python manage.py showmigrations

# Revertir migración
python manage.py migrate accounts 0001
```

---

## Funcionalidades Principales

### 1. Preprocesamiento Automático
- **Encoding de variables categóricas**: LabelEncoder automático
- **Escalado de características**: StandardScaler para normalización
- **Manejo de valores nulos**: Imputación con media/moda
- **Detección de tipo de problema**: Clasificación vs Regresión

### 2. Sistema de Plantillas Personalizadas
Cada perfil puede tener su propia plantilla de resultados:
- **Genérica**: `resultados.html` (fallback)
- **Perfil 1**: `resultados_perfil1.html` (Pobreza Puno)
- **Perfil 4**: `resultados_perfil4.html` (Rendimiento Académico)
- **Perfil 5**: `resultados_perfil5.html` (Anemia Gestantes)

Lógica en `views.py`:
```python
def dashboard_resultados(request, perfil_id):
    perfil = get_object_or_404(Perfil, id=perfil_id)

    # Buscar plantilla personalizada
    template_name = f'dashboard/resultados_perfil{perfil.id}.html'

    # Fallback a genérica
    if not template_exists(template_name):
        template_name = 'dashboard/resultados.html'

    return render(request, template_name, {'perfil': perfil})
```

### 3. Interpretación Contextualizada de Métricas

#### Para Regresión (Perfil 4 - Noemi)
- R² Score con niveles de calidad:
  - ≥ 0.7: "Excelente capacidad explicativa"
  - ≥ 0.5: "Capacidad moderada"
  - < 0.5: "Capacidad limitada"
- RMSE con contexto educativo:
  - "Error típico de ±0.65 puntos en la calificación"
- Ejemplo práctico:
  - "Si el modelo predice 15.0, la nota real estará entre 14.4 y 15.6"

#### Para Clasificación Multiclase (Perfil 5 - Zulema)
- Accuracy con evaluación clínica:
  - ≥ 0.9: "Excelente para uso clínico"
  - ≥ 0.8: "Muy bueno para detección"
  - < 0.8: "Requiere mejoras"
- Matriz de Confusión 4x4 con análisis detallado:
  - Accuracy por clase (Normal: 97%, Leve: 96%, Moderada: 99%, Severa: 93%)
  - Detección de errores críticos (ej: 20 casos severos mal clasificados)
  - Impacto en salud pública con números específicos

### 4. Validación Cruzada (Cross-Validation)
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

metricas = {
    'CV_Mean_Score': float(cv_scores.mean()),
    'CV_Std_Score': float(cv_scores.std()),
}
```

### 5. Seguridad
- **Autenticación por PIN**: No requiere contraseñas complejas
- **Validación de códigos de acceso**: Control de registro
- **CSRF Protection**: Protección contra ataques cross-site
- **Validación de archivos**: Solo CSVs, máximo 10 MB
- **Sanitización de datos**: Prevención de inyección SQL

---

## Despliegue

### Opción 1: Despliegue con Docker

#### 1. Construir imagen
```bash
docker build -t supervisado-dashboard .
```

#### 2. Ejecutar con Docker Compose
```bash
docker-compose up -d
```

#### 3. Acceder al sistema
```
http://localhost:8000
```

### Opción 2: Despliegue en Servidor Linux

Ver guía completa en [DESPLIEGUE.md](DESPLIEGUE.md)

**Pasos resumidos:**
1. Instalar Python 3.8+
2. Configurar entorno virtual
3. Instalar dependencias
4. Configurar PostgreSQL
5. Configurar Gunicorn
6. Configurar Nginx
7. Configurar variables de entorno
8. Ejecutar migraciones
9. Recolectar archivos estáticos
10. Iniciar servicios

---

## Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# Django
SECRET_KEY=tu-clave-secreta-muy-larga-y-segura
DEBUG=False
ALLOWED_HOSTS=tu-dominio.com,www.tu-dominio.com

# Base de Datos
DATABASE_URL=postgresql://usuario:password@localhost:5432/supervisado_db

# Archivos Estáticos
STATIC_URL=/static/
MEDIA_URL=/media/

# Email (opcional)
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=tu-email@gmail.com
EMAIL_HOST_PASSWORD=tu-password

# Seguridad
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
```

---

## Dependencias Principales

Archivo `requirements.txt`:

```
Django==5.2.7
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
Pillow==10.1.0
gunicorn==21.2.0
whitenoise==6.6.0
psycopg2-binary==2.9.9
python-decouple==3.8
```

---

## Testing

### Ejecutar tests
```bash
python manage.py test
```

### Tests por aplicación
```bash
python manage.py test apps.accounts
python manage.py test apps.dashboard
python manage.py test apps.ml_models
```

---

## Solución de Problemas

### Error: "Classification metrics can't handle a mix of multiclass and continuous targets"
**Solución**: El sistema ahora mapea explícitamente algoritmos a tipos de problema:
```python
algoritmos_regresion = ['regresion_lineal', 'regresion_polinomial', 'ridge', 'lasso']
algoritmos_clasificacion = ['regresion_logistica', 'decision_tree', 'random_forest', 'svm', 'knn']
```

### Error: "TemplateSyntaxError: Invalid filter 'mul'"
**Solución**: Django no tiene filtro `mul` nativo. Usar `widthratio` o custom template tags.

### Error al cargar CSV
**Solución**:
- Verificar encoding (UTF-8 o Latin-1)
- Última columna debe ser el target
- No debe haber filas completamente vacías
- Tamaño máximo: 10 MB

### R² Score se muestra incorrectamente (0.45% en vez de 44%)
**Solución**: Usar template tag `widthratio`:
```django
{% widthratio perfil.metricas.R2_Score 1 100 as r2_pct %}
{{ r2_pct }}%
```

---

## Roadmap Futuro

### Versión 2.0
- [ ] Exportación de resultados a PDF
- [ ] Descarga del modelo entrenado (.pkl)
- [ ] Comparación de múltiples modelos lado a lado
- [ ] Gráficos interactivos con Plotly
- [ ] Notebook Jupyter embebido

### Versión 3.0
- [ ] API REST con Django REST Framework
- [ ] Frontend con React/Vue
- [ ] Sistema de roles (Admin, Docente, Estudiante)
- [ ] Chat en tiempo real para soporte
- [ ] Integración con Google Classroom

### Mejoras Continuas
- [ ] Más algoritmos (XGBoost, LightGBM, CatBoost)
- [ ] AutoML (búsqueda automática de hiperparámetros)
- [ ] Detección de outliers
- [ ] Feature engineering automático
- [ ] Explicabilidad con SHAP/LIME

---

## Contribuciones

Este es un proyecto educativo. Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crear rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abrir Pull Request

### Guía de Estilo
- Seguir PEP 8 para Python
- Docstrings en todas las funciones
- Tests para nuevas funcionalidades
- Commits descriptivos en español

---

## Licencia

Proyecto Educativo - Universidad Nacional del Altiplano
Facultad de Ingeniería Estadística e Informática
Ciclo VII - 2025

Uso libre para fines académicos y educativos.

---

## Autores

**Proyecto desarrollado por estudiantes de Ciclo VII:**
- **Cliver** - Perfil 1: Predicción de Pobreza en Puno
- **Noemi** - Perfil 4: Rendimiento Académico Estudiantil
- **Zulema** - Perfil 5: Clasificación de Anemia en Gestantes

**Docente**: [Nombre del profesor]

---

## Contacto y Soporte

- **Issues**: Reportar bugs en GitHub Issues
- **Email**: [correo del equipo]
- **Documentación**: Este README.md

---

## Changelog

### v1.0.0 (2025-01-15)
- Sistema base con Django 5.2.7
- Autenticación con PIN
- 10 algoritmos de ML soportados
- Sistema de perfiles personalizados
- Validación cruzada
- Plantillas personalizadas por perfil

### v1.1.0 (2025-01-20)
- Fix: Mapeo explícito de algoritmos a tipos de problema
- Fix: Corrección de display de R² Score
- Mejora: Interpretación contextualizada de métricas
- Mejora: Análisis detallado de matriz de confusión 4x4
- Mejora: Explicaciones mejoradas de validación cruzada

---

**Desarrollado con Django y scikit-learn para el aprendizaje de Machine Learning**

*Ciclo VII - Facultad de Ingeniería Estadística e Informática - 2025*
