from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from apps.accounts.models import PerfilEstudiante

def landing_page(request):
    """Vista de la landing page con todos los perfiles"""
    
    perfiles = PerfilEstudiante.objects.filter(activo=True).order_by('-fecha_registro')
    
    context = {
        'perfiles': perfiles
    }
    return render(request, 'dashboard/landing.html', context)


def dashboard_estudiante(request, perfil_id):
    """Dashboard personal del estudiante - redirige a la sección inicio"""

    # Verificar que tenga acceso (PIN correcto en sesión)
    perfil_activo = request.session.get('perfil_activo')

    if perfil_activo != perfil_id:
        messages.warning(request, 'Debes ingresar el PIN para acceder a este proyecto')
        return redirect('login_pin', perfil_id=perfil_id)

    # Redirigir directamente a la sección inicio del dashboard
    return redirect('dashboard_inicio', perfil_id=perfil_id)


def seleccionar_algoritmo(request, perfil_id):
    """Vista para seleccionar el algoritmo a usar"""
    
    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)
    
    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)
    
    if request.method == 'POST':
        algoritmo = request.POST.get('algoritmo')
        perfil.algoritmo_elegido = algoritmo
        perfil.save()
        
        messages.success(request, f'Algoritmo seleccionado: {perfil.get_algoritmo_elegido_display()}')
        return redirect('dashboard_estudiante', perfil_id=perfil_id)
    
    context = {
        'perfil': perfil,
        'algoritmos': PerfilEstudiante.ALGORITMOS_CHOICES
    }
    return render(request, 'dashboard/seleccionar_algoritmo.html', context)


def dashboard_inicio(request, perfil_id):
    """Vista del dashboard principal - sección inicio"""

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        messages.warning(request, 'Debes ingresar el PIN para acceder a este proyecto')
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    # Calcular progreso
    progreso = 25  # Base
    if perfil.dataset_nombre:
        progreso += 25
    if perfil.algoritmo_elegido:
        progreso += 25
    if perfil.tiene_modelo_entrenado():
        progreso += 25

    context = {
        'perfil': perfil,
        'progreso': progreso
    }
    return render(request, 'dashboard/inicio.html', context)


def dashboard_datos(request, perfil_id):
    """Vista de gestión de datos"""
    import pandas as pd
    import os

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    if request.method == 'POST':
        # Verificar si se quiere eliminar el dataset
        if request.POST.get('eliminar_dataset'):
            # Eliminar el archivo físico si existe
            if perfil.dataset_archivo:
                perfil.dataset_archivo.delete(save=False)

            # Limpiar los campos en la base de datos
            perfil.dataset_archivo = None
            perfil.dataset_nombre = None
            perfil.save()

            messages.success(request, 'Dataset eliminado. Puedes cargar uno nuevo.')
            return redirect('dashboard_datos', perfil_id=perfil_id)

        # Procesar archivo subido
        archivo = request.FILES.get('dataset_archivo')
        if archivo:
            # Validar que sea CSV
            if not archivo.name.endswith('.csv'):
                messages.error(request, 'Solo se permiten archivos CSV')
                return redirect('dashboard_datos', perfil_id=perfil_id)

            # Si ya existe un dataset, eliminarlo primero
            if perfil.dataset_archivo:
                perfil.dataset_archivo.delete(save=False)

            # Guardar el archivo
            perfil.dataset_archivo = archivo
            perfil.dataset_nombre = archivo.name
            perfil.save()

            messages.success(request, f'Dataset "{archivo.name}" cargado exitosamente')
            return redirect('dashboard_datos', perfil_id=perfil_id)

    # Si existe un dataset, leer y mostrar preview
    context = {
        'perfil': perfil,
        'dataset_info': None,
        'preview_data': None,
        'preview_columns': None
    }

    if perfil.dataset_archivo:
        try:
            # Intentar leer el archivo CSV con diferentes encodings
            df = None
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    df = pd.read_csv(perfil.dataset_archivo.path, encoding=encoding)
                    break  # Si funciona, salir del loop
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise Exception("No se pudo decodificar el archivo con ningún encoding conocido")

            # Información básica
            context['dataset_info'] = {
                'rows': len(df),
                'columns': len(df.columns)
            }

            # Primeras 5 filas
            preview_df = df.head(5)
            context['preview_columns'] = list(df.columns)
            context['preview_data'] = preview_df.values.tolist()

        except Exception as e:
            messages.error(request, f'Error al leer el dataset: {str(e)}')

    return render(request, 'dashboard/datos.html', context)


def dashboard_modelo(request, perfil_id):
    """Vista de configuración del modelo"""
    import pandas as pd
    import pickle
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import (
        mean_squared_error, r2_score, accuracy_score,
        confusion_matrix, classification_report,
        roc_curve, auc, roc_auc_score
    )
    from sklearn.preprocessing import StandardScaler
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    if request.method == 'POST':
        # Guardar algoritmo seleccionado
        algoritmo = request.POST.get('algoritmo')
        if algoritmo:
            perfil.algoritmo_elegido = algoritmo
            perfil.save()
            messages.success(request, f'Algoritmo seleccionado: {perfil.get_algoritmo_elegido_display()}')
            return redirect('dashboard_modelo', perfil_id=perfil_id)

        # Entrenar modelo
        if request.POST.get('entrenar') and perfil.dataset_archivo:
            try:
                # Leer dataset con diferentes encodings
                df = None
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(perfil.dataset_archivo.path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue

                if df is None:
                    raise Exception("No se pudo leer el archivo CSV")

                # Preparar datos (última columna es target)
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                # Eliminar filas con valores nulos en y (target)
                mask = y.notna()
                X = X[mask]
                y = y[mask]

                # Convertir columnas categóricas a numéricas
                from sklearn.preprocessing import LabelEncoder
                label_encoders = {}
                for col in X.columns:
                    if X[col].dtype == 'object':  # Si es texto
                        le = LabelEncoder()
                        X[col] = X[col].fillna('missing')
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le

                # Rellenar valores nulos numéricos con la mediana
                X = X.fillna(X.median())

                # Detectar si es clasificación (target categórico o binario)
                is_classification = y.dtype == 'object' or len(np.unique(y)) <= 10

                # Si y es categórica, convertirla
                le_y = None
                if y.dtype == 'object':
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y.astype(str))
                    is_classification = True

                # Split de datos
                test_size = int(request.POST.get('test_size', 30)) / 100
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Escalar datos
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Seleccionar y entrenar modelo según algoritmo
                if perfil.algoritmo_elegido == 'regresion_lineal':
                    model = LinearRegression()
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                elif perfil.algoritmo_elegido == 'regresion_logistica':
                    model = LogisticRegression(max_iter=1000)
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                    is_classification = True
                elif perfil.algoritmo_elegido == 'ridge_lasso':
                    model = Ridge(alpha=1.0)
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                elif perfil.algoritmo_elegido == 'arbol_cart':
                    model = DecisionTreeRegressor()
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                elif perfil.algoritmo_elegido == 'knn':
                    model = KNeighborsRegressor(n_neighbors=5)
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                elif perfil.algoritmo_elegido == 'red_neuronal':
                    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
                    X_train_final = X_train_scaled
                    X_test_final = X_test_scaled
                else:
                    raise Exception("Algoritmo no válido")

                # Entrenar
                model.fit(X_train_final, y_train)

                # Predecir
                y_pred = model.predict(X_test_final)

                # Inicializar métricas
                metricas = {
                    'Train Size': len(X_train),
                    'Test Size': len(X_test),
                    'Num Features': X.shape[1]
                }

                # Calcular métricas según tipo de problema
                if is_classification and hasattr(model, 'predict_proba'):
                    # Clasificación con probabilidades
                    y_pred_class = np.round(y_pred).astype(int)

                    # Accuracy
                    accuracy = accuracy_score(y_test, y_pred_class)
                    metricas['Accuracy'] = float(accuracy)

                    # Matriz de confusión
                    cm = confusion_matrix(y_test, y_pred_class)
                    metricas['Confusion_Matrix'] = cm.tolist()

                    # ROC y AUC si es binario
                    if len(np.unique(y)) == 2:
                        try:
                            y_proba = model.predict_proba(X_test_final)[:, 1]
                            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                            roc_auc = auc(fpr, tpr)

                            metricas['AUC'] = float(roc_auc)
                            metricas['ROC_FPR'] = fpr.tolist()
                            metricas['ROC_TPR'] = tpr.tolist()
                        except:
                            pass

                    # MSE y R2 también
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    metricas['MSE'] = float(mse)
                    metricas['R2 Score'] = float(r2)

                else:
                    # Regresión
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    metricas['MSE'] = float(mse)
                    metricas['RMSE'] = float(rmse)
                    metricas['R2 Score'] = float(r2)
                    metricas['MAE'] = float(np.mean(np.abs(y_test - y_pred)))

                # Feature importance (si está disponible)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = list(X.columns)
                    metricas['Feature_Importance'] = {
                        'features': feature_names,
                        'importances': importances.tolist()
                    }
                elif hasattr(model, 'coef_'):
                    coefs = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                    feature_names = list(X.columns)
                    metricas['Feature_Coefficients'] = {
                        'features': feature_names,
                        'coefficients': coefs.tolist()
                    }

                # Guardar nombre de variable objetivo
                metricas['target_column'] = df.columns[-1]
                metricas['feature_columns'] = list(X.columns)

                # Guardar métricas
                perfil.metricas = metricas

                # Guardar modelo y datos adicionales
                modelo_path = os.path.join('media', 'modelos', f'modelo_{perfil.id}.pkl')
                os.makedirs(os.path.dirname(modelo_path), exist_ok=True)

                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'label_encoders': label_encoders,
                    'le_y': le_y,
                    'feature_names': list(X.columns),
                    'is_classification': is_classification,
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist()
                }

                with open(modelo_path, 'wb') as f:
                    pickle.dump(model_data, f)

                perfil.modelo_archivo = modelo_path.replace('media/', '')
                perfil.save()

                messages.success(request, 'Modelo entrenado exitosamente')

            except Exception as e:
                import traceback
                messages.error(request, f'Error al entrenar el modelo: {str(e)}')
                print(traceback.format_exc())

            return redirect('dashboard_modelo', perfil_id=perfil_id)

    context = {
        'perfil': perfil,
        'algoritmos': PerfilEstudiante.ALGORITMOS_CHOICES
    }
    return render(request, 'dashboard/modelo.html', context)


def dashboard_resultados(request, perfil_id):
    """Vista de resultados y métricas"""

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    context = {
        'perfil': perfil
    }
    return render(request, 'dashboard/resultados.html', context)


def dashboard_config(request, perfil_id):
    """Vista de configuración del proyecto"""

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    if request.method == 'POST':
        # Actualizar configuración
        perfil.nombre_completo = request.POST.get('nombre_completo', perfil.nombre_completo)
        perfil.save()
        messages.success(request, 'Configuración actualizada exitosamente')
        return redirect('dashboard_config', perfil_id=perfil_id)

    context = {
        'perfil': perfil
    }
    return render(request, 'dashboard/config.html', context)