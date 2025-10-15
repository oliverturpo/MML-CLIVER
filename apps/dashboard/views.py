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

    # Seleccionar template según perfil_id
    if perfil_id == 4:
        template_name = 'dashboard/inicio_perfil4.html'
    elif perfil_id == 5:
        template_name = 'dashboard/inicio_perfil5.html'
    else:
        template_name = 'dashboard/inicio.html'

    return render(request, template_name, context)


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

    # Si existe un dataset, leer y mostrar preview + análisis EDA
    context = {
        'perfil': perfil,
        'dataset_info': None,
        'preview_data': None,
        'preview_columns': None,
        'eda_analisis': None  # Nuevo: análisis exploratorio
    }

    if perfil.dataset_archivo:
        try:
            # ========== USAR CLASES POO PARA ANÁLISIS ==========
            from apps.ml_models import DataLoader, DataPreprocessor
            import numpy as np

            # 1. CARGAR DATOS CON DataLoader (POO)
            loader = DataLoader()
            loader.load_csv(perfil.dataset_archivo.path)
            df = loader.get_data()

            # Información básica
            context['dataset_info'] = {
                'rows': len(df),
                'columns': len(df.columns)
            }

            # Primeras 5 filas
            preview_df = df.head(5)
            context['preview_columns'] = list(df.columns)
            context['preview_data'] = preview_df.values.tolist()

            # ========== ANÁLISIS EXPLORATORIO DE DATOS (EDA) ==========
            eda_analisis = {}

            # 1. TIPOS DE VARIABLES
            X = df.iloc[:, :-1]  # Features
            y = df.iloc[:, -1]   # Target

            numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categoricas = X.select_dtypes(include=['object']).columns.tolist()

            eda_analisis['tipos_variables'] = {
                'numericas': numericas,
                'categoricas': categoricas,
                'total_numericas': len(numericas),
                'total_categoricas': len(categoricas)
            }

            # 2. VALORES FALTANTES
            valores_faltantes = df.isnull().sum()
            porcentaje_faltantes = (valores_faltantes / len(df) * 100).round(2)

            faltantes_info = []
            for col in df.columns:
                if valores_faltantes[col] > 0:
                    faltantes_info.append({
                        'columna': col,
                        'cantidad': int(valores_faltantes[col]),
                        'porcentaje': float(porcentaje_faltantes[col])
                    })

            eda_analisis['valores_faltantes'] = {
                'total_columnas_con_faltantes': len(faltantes_info),
                'detalles': faltantes_info,
                'tiene_faltantes': len(faltantes_info) > 0
            }

            # 3. ESTADÍSTICAS DESCRIPTIVAS (solo numéricas)
            if len(numericas) > 0:
                desc = df[numericas].describe().round(2)

                estadisticas = []
                for col in numericas:
                    estadisticas.append({
                        'columna': col,
                        'count': int(desc.loc['count', col]),
                        'mean': float(desc.loc['mean', col]),
                        'std': float(desc.loc['std', col]),
                        'min': float(desc.loc['min', col]),
                        'q25': float(desc.loc['25%', col]),
                        'median': float(desc.loc['50%', col]),
                        'q75': float(desc.loc['75%', col]),
                        'max': float(desc.loc['max', col])
                    })

                eda_analisis['estadisticas_numericas'] = estadisticas

            # 4. DESBALANCEO EN VARIABLE OBJETIVO
            y_counts = y.value_counts()
            y_percentages = (y.value_counts(normalize=True) * 100).round(2)

            distribuciones = []
            for valor in y_counts.index:
                distribuciones.append({
                    'valor': str(valor),
                    'cantidad': int(y_counts[valor]),
                    'porcentaje': float(y_percentages[valor])
                })

            # Detectar desbalanceo severo (si una clase < 30%)
            desbalanceado = any(p < 30 for p in y_percentages.values) if len(y_percentages) == 2 else False

            eda_analisis['distribucion_target'] = {
                'nombre_variable': df.columns[-1],
                'clases': distribuciones,
                'total_clases': len(distribuciones),
                'desbalanceado': desbalanceado,
                'es_binario': len(distribuciones) == 2
            }

            # 5. VALORES ATÍPICOS (OUTLIERS) - Método IQR
            outliers_info = []
            for col in numericas:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

                if len(outliers) > 0:
                    outliers_info.append({
                        'columna': col,
                        'cantidad': len(outliers),
                        'porcentaje': round((len(outliers) / len(df)) * 100, 2),
                        'rango_esperado': f'[{lower_bound:.2f}, {upper_bound:.2f}]'
                    })

            eda_analisis['outliers'] = {
                'total_columnas_con_outliers': len(outliers_info),
                'detalles': outliers_info,
                'tiene_outliers': len(outliers_info) > 0
            }

            # 6. RESUMEN DE DESAFÍOS
            desafios = []
            if eda_analisis['valores_faltantes']['tiene_faltantes']:
                desafios.append('Valores faltantes detectados')
            if eda_analisis['distribucion_target']['desbalanceado']:
                desafios.append('Desbalanceo de clases en variable objetivo')
            if eda_analisis['outliers']['tiene_outliers']:
                desafios.append('Valores atípicos (outliers) detectados')

            eda_analisis['desafios_detectados'] = desafios
            eda_analisis['tiene_desafios'] = len(desafios) > 0

            context['eda_analisis'] = eda_analisis

        except Exception as e:
            import traceback
            messages.error(request, f'Error al analizar el dataset: {str(e)}')
            print(traceback.format_exc())

    return render(request, 'dashboard/datos.html', context)


def dashboard_modelo(request, perfil_id):
    """
    Vista de configuración y entrenamiento del modelo usando clases POO.

    Versión POO que reemplaza 250+ líneas de código inline con
    llamadas limpias a las clases POO del sistema ML.
    """
    import pandas as pd
    import os

    # Validación de acceso
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

        # Entrenar modelo con POO
        if request.POST.get('entrenar') and perfil.dataset_archivo:
            try:
                # Importar clases POO
                from apps.ml_models import MLPipeline

                # Detectar tipo de problema automáticamente
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

                # Determinar tipo de problema según el algoritmo elegido
                # Forzar el tipo correcto para evitar conflictos
                algoritmos_regresion = ['regresion_lineal', 'regresion_polinomial', 'ridge', 'lasso']
                algoritmos_clasificacion = ['regresion_logistica', 'decision_tree', 'random_forest', 'svm', 'knn']

                if perfil.algoritmo_elegido in algoritmos_regresion:
                    problem_type = 'regression'
                elif perfil.algoritmo_elegido in algoritmos_clasificacion:
                    problem_type = 'classification'
                else:
                    # Detección automática como respaldo
                    y = df.iloc[:, -1]
                    is_classification = y.dtype == 'object' or len(y.unique()) <= 10
                    problem_type = 'classification' if is_classification else 'regression'

                # Crear y ejecutar pipeline POO
                pipeline = MLPipeline(
                    algorithm=perfil.algoritmo_elegido,
                    problem_type=problem_type
                )

                test_size = int(request.POST.get('test_size', 30)) / 100

                # Ejecutar pipeline completo con POO
                # Esto reemplaza 150+ líneas de código inline
                results = pipeline.run_complete_pipeline(
                    file_path=perfil.dataset_archivo.path,
                    test_size=test_size,
                    val_size=0.1,
                    use_cross_validation=True,      # Validación cruzada
                    hyperparameter_tuning=True,      # Ajuste de hiperparámetros
                    engineer_features=False
                )

                # Extraer métricas del resultado
                evaluation_metrics = results.get('evaluation', {})

                # Agregar información adicional
                if 'pipeline_config' in results:
                    config = results['pipeline_config']
                    evaluation_metrics['Train_Size'] = config['n_samples'] - int(config['n_samples'] * test_size)
                    evaluation_metrics['Test_Size'] = int(config['n_samples'] * test_size)
                    evaluation_metrics['Num_Features'] = config['n_features']

                # Agregar información de validación cruzada
                if 'training' in results and 'cross_validation' in results['training']:
                    cv_results = results['training']['cross_validation']
                    evaluation_metrics['CV_Mean_Score'] = cv_results['mean_score']
                    evaluation_metrics['CV_Std_Score'] = cv_results['std_score']

                # Agregar mejores hiperparámetros
                if 'training' in results and 'hyperparameter_tuning' in results['training']:
                    tuning_results = results['training']['hyperparameter_tuning']
                    evaluation_metrics['Best_Params'] = tuning_results['best_params']
                    evaluation_metrics['Best_CV_Score'] = tuning_results['best_score']

                # Agregar feature importance/coefficients
                feature_info = pipeline.get_feature_importance()
                if feature_info:
                    if 'importances' in feature_info:
                        evaluation_metrics['Feature_Importance'] = feature_info
                    elif 'coefficients' in feature_info:
                        evaluation_metrics['Feature_Coefficients'] = feature_info

                # Convertir todo a tipos JSON-serializables y validar
                import json
                import numpy as np
                import math

                def make_json_serializable(obj):
                    """
                    Convierte objetos numpy y otros a tipos JSON-serializables.
                    Maneja también NaN, Inf y otros casos especiales.
                    """
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        val = float(obj)
                        # Manejar NaN, Inf, -Inf
                        if math.isnan(val) or math.isinf(val):
                            return None
                        return val
                    elif isinstance(obj, float):
                        # Manejar float de Python también
                        if math.isnan(obj) or math.isinf(obj):
                            return None
                        return obj
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: make_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_serializable(item) for item in obj]
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif obj is None:
                        return None
                    elif isinstance(obj, (str, int, bool)):
                        return obj
                    else:
                        # Intentar convertir a string como último recurso
                        return str(obj)

                evaluation_metrics = make_json_serializable(evaluation_metrics)

                # Validar que el JSON es serializable antes de guardar
                try:
                    json_test = json.dumps(evaluation_metrics)
                    # Si llegamos aquí, es válido
                except (TypeError, ValueError) as json_error:
                    raise Exception(f"Error al serializar métricas a JSON: {str(json_error)}")

                # Guardar métricas en el perfil
                perfil.metricas = evaluation_metrics

                # Guardar pipeline completo
                modelo_path = os.path.join('media', 'modelos', f'pipeline_{perfil.id}.pkl')
                os.makedirs(os.path.dirname(modelo_path), exist_ok=True)

                pipeline.save_pipeline(modelo_path)

                perfil.modelo_archivo = modelo_path.replace('media/', '')
                perfil.save()

                # Mensaje de éxito
                messages.success(
                    request,
                    f'Modelo entrenado exitosamente con {perfil.get_algoritmo_elegido_display()}. '
                    f'CV Score: {evaluation_metrics.get("CV_Mean_Score", 0):.4f}'
                )

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

    # Seleccionar template según perfil_id
    if perfil_id == 4:
        template_name = 'dashboard/resultados_perfil4.html'
    elif perfil_id == 5:
        template_name = 'dashboard/resultados_perfil5.html'
    else:
        template_name = 'dashboard/resultados.html'

    return render(request, template_name, context)


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


def eliminar_perfil(request, perfil_id):
    """Vista para eliminar completamente el perfil del estudiante"""

    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    if request.method == 'POST':
        # Eliminar archivos asociados
        if perfil.dataset_archivo:
            perfil.dataset_archivo.delete(save=False)
        if perfil.modelo_archivo:
            perfil.modelo_archivo.delete(save=False)
        if perfil.avatar:
            perfil.avatar.delete(save=False)

        nombre = perfil.nombre_completo

        # Eliminar perfil de la base de datos
        perfil.delete()

        # Limpiar sesión
        if 'perfil_activo' in request.session:
            del request.session['perfil_activo']

        messages.success(request, f'Perfil de {nombre} eliminado exitosamente')
        return redirect('landing_page')

    # Si es GET, redirigir a config
    return redirect('dashboard_config', perfil_id=perfil_id)