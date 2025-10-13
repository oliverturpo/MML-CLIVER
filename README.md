# 📊 Dashboard de Aprendizaje Supervisado

Plataforma educativa interactiva para estudiantes de Machine Learning. Permite crear perfiles individuales, entrenar modelos y visualizar resultados con métricas avanzadas.

## 🚀 Características Principales

### Sistema de Usuarios
- ✅ **Registro con código de acceso** (controlado por el docente)
- ✅ **Login con PIN de 4 dígitos** (sin sistema tradicional de usuarios)
- ✅ **Perfiles personalizados** con avatar opcional

### Algoritmos Soportados
1. **Regresión Lineal**
2. **Regresión Logística** (con Curva ROC y AUC)
3. **Ridge y Lasso**
4. **Árbol de Regresión CART**
5. **K-Nearest Neighbors (KNN)**
6. **Red Neuronal para Regresión**

### Visualizaciones Avanzadas
- 📈 **Curva ROC** con cálculo de AUC (para clasificación binaria)
- 📊 **Matriz de Confusión** (clasificación)
- 🎯 **Importancia de Características** / Coeficientes
- 📉 **Métricas de Rendimiento** (MSE, RMSE, R², MAE, Accuracy)

## 🛠️ Tecnologías Utilizadas

**Backend:**
- Django 5.2.7
- Python 3.x

**Machine Learning:**
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- plotly

**Frontend:**
- Tailwind CSS
- Chart.js
- Font Awesome
- Alpine.js (opcional)

**Base de Datos:**
- SQLite (desarrollo)

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd SUPERVISADO-DASHBOARD
```

### 2. Crear entorno virtual
```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Realizar migraciones
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Crear superusuario (opcional)
```bash
python manage.py createsuperuser
```

### 7. Ejecutar el servidor
```bash
python manage.py runserver
```

### 8. Acceder a la aplicación
Abre tu navegador en: `http://127.0.0.1:8000/`

## 📚 Uso del Sistema

### Para Estudiantes

#### 1. **Registro**
- Ve a la página de registro
- Ingresa tu nombre completo, email, carrera
- Crea un PIN de 4 dígitos (recuérdalo!)
- Usa el código de acceso proporcionado por tu docente

#### 2. **Acceso al Dashboard**
- En la landing page, busca tu perfil
- Haz clic en "Ver Proyecto"
- Ingresa tu PIN de 4 dígitos

#### 3. **Cargar Dataset**
- Ve a la sección "Datos"
- Sube tu archivo CSV
  - **Última columna = variable objetivo (target)**
  - Puede contener datos categóricos (se convertirán automáticamente)
- Visualiza el preview del dataset

#### 4. **Seleccionar Algoritmo**
- Ve a la sección "Modelo"
- Selecciona uno de los 6 algoritmos disponibles
- Configura el tamaño del conjunto de prueba (20-30% recomendado)

#### 5. **Entrenar Modelo**
- Haz clic en "Entrenar Modelo"
- El sistema automáticamente:
  - Preprocesa los datos
  - Entrena el modelo
  - Calcula métricas
  - Genera visualizaciones

#### 6. **Ver Resultados**
- Ve a la sección "Resultados"
- Observa:
  - Métricas principales (R², MSE, RMSE, Accuracy, AUC)
  - Curva ROC (clasificación binaria)
  - Matriz de confusión (clasificación)
  - Importancia de características
  - Coeficientes del modelo

### Para Docentes

#### 1. **Gestión de Códigos de Acceso**
- Accede al admin: `http://127.0.0.1:8000/admin/`
- Ve a "Códigos de Acceso"
- Crea códigos con:
  - Límite de usos
  - Fecha de expiración (opcional)
  - Descripción

#### 2. **Supervisión de Estudiantes**
- Ve a "Perfiles de Estudiantes"
- Observa el progreso de cada estudiante
- Revisa sus modelos y métricas

## 📊 Datasets de Ejemplo

El proyecto incluye 3 datasets de ejemplo:

### 1. `ejemplo_dataset.csv`
- Dataset básico con datos de empleados
- Columnas: edad, salario, experiencia, departamento, calificación
- Perfecto para regresión lineal

### 2. `diabetes.csv`
- Dataset clásico de diabetes
- 768 registros con 8 características médicas
- Target binario: Outcome (0 o 1)
- **Ideal para Regresión Logística y Curva ROC**

### 3. `titanic.csv`
- Dataset del Titanic
- Clasificación binaria: Survived (0 o 1)
- **Perfecto para Regresión Logística**

## 🔑 Casos de Uso

### Ejemplo 1: Regresión Logística con Diabetes
```python
1. Subir diabetes.csv
2. Seleccionar "Regresión Logística"
3. Entrenar con 30% de test
4. Ver Curva ROC con AUC
5. Analizar matriz de confusión
```

### Ejemplo 2: Regresión Ridge
```python
1. Subir ejemplo_dataset.csv
2. Seleccionar "Ridge y Lasso"
3. Entrenar el modelo
4. Ver coeficientes regularizados
```

### Ejemplo 3: Árbol de Decisión
```python
1. Subir titanic.csv
2. Seleccionar "Árbol de Regresión CART"
3. Ver importancia de características
```

## 🎨 Características Especiales

### Preprocesamiento Automático
- ✅ Conversión de variables categóricas a numéricas
- ✅ Manejo de valores faltantes
- ✅ Escalado de características (StandardScaler)
- ✅ Detección automática de tipo de problema (clasificación/regresión)

### Visualizaciones Dinámicas
- ✅ **Curva ROC** real con datos del modelo
- ✅ **Matriz de Confusión** calculada
- ✅ **Gráficos interactivos** con Chart.js
- ✅ **Feature Importance** para árboles
- ✅ **Coeficientes** para modelos lineales

### Seguridad
- ✅ Acceso por PIN (sin contraseñas)
- ✅ Validación de códigos de acceso
- ✅ Sesiones seguras
- ✅ Protección CSRF

## 📱 Responsive Design
- ✅ Diseño adaptable a móviles
- ✅ Sidebar colapsable
- ✅ Gráficos responsivos
- ✅ Optimizado para tablets

## 🎯 Roadmap Futuro

- [ ] Exportar resultados a PDF
- [ ] Descargar modelo entrenado
- [ ] Comparación de múltiples modelos
- [ ] Validación cruzada
- [ ] Más algoritmos (SVM, Random Forest, XGBoost)
- [ ] Notebook interactivo

## 🐛 Solución de Problemas

### Error al cargar CSV
- Asegúrate que el archivo sea CSV válido
- La última columna debe ser el target
- Usa encoding UTF-8 o Latin-1

### Modelo no entrena
- Verifica que el dataset tenga datos suficientes
- Mínimo recomendado: 50 registros
- Revisa que no haya columnas vacías

### Curva ROC no aparece
- Solo disponible para clasificación binaria
- El target debe tener exactamente 2 clases
- Usa Regresión Logística

## 👥 Contribuir

Este es un proyecto educativo. Si quieres contribuir:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📄 Licencia

Proyecto educativo - Uso libre para fines académicos

## 📧 Contacto

Para dudas o sugerencias sobre el proyecto, contacta al equipo de desarrollo.

---

**🤖 Desarrollado con Django y Machine Learning para el aprendizaje supervisado**

*Ciclo VII - Facultad de Ingeniería Estadística e Informática*
