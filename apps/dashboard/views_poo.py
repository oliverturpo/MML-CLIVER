"""
Versión MEJORADA de dashboard_modelo usando Programación Orientada a Objetos

Este archivo contiene la versión refactorizada del método dashboard_modelo
que usa las clases POO del módulo ml_models.

ANTES: 250+ líneas de código inline mezclado
DESPUÉS: ~80 líneas usando clases POO

Principios POO demostrados:
- Encapsulamiento
- Separación de responsabilidades
- Composición
- Reutilización de código
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from apps.accounts.models import PerfilEstudiante
import os
import pickle


def dashboard_modelo_poo(request, perfil_id):
    """
    Vista de configuración y entrenamiento del modelo usando clases POO.

    Esta versión reemplaza 250+ líneas de código inline con
    llamadas limpias a las clases POO del sistema.
    """
    # ========================================================================
    # VALIDACIÓN DE ACCESO (igual que antes)
    # ========================================================================
    perfil_activo = request.session.get('perfil_activo')
    if perfil_activo != perfil_id:
        return redirect('login_pin', perfil_id=perfil_id)

    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)

    # ========================================================================
    # MANEJO DE PETICIONES POST
    # ========================================================================
    if request.method == 'POST':

        # --------------------------------------------------------------------
        # GUARDAR ALGORITMO SELECCIONADO
        # --------------------------------------------------------------------
        algoritmo = request.POST.get('algoritmo')
        if algoritmo:
            perfil.algoritmo_elegido = algoritmo
            perfil.save()
            messages.success(request, f'Algoritmo seleccionado: {perfil.get_algoritmo_elegido_display()}')
            return redirect('dashboard_modelo', perfil_id=perfil_id)

        # --------------------------------------------------------------------
        # ENTRENAR MODELO CON CLASES POO ✨
        # --------------------------------------------------------------------
        if request.POST.get('entrenar') and perfil.dataset_archivo:
            try:
                # Importar clases POO
                from apps.ml_models import MLPipeline

                # ============================================================
                # DETECTAR TIPO DE PROBLEMA AUTOMÁTICAMENTE
                # ============================================================
                import pandas as pd

                # Leer dataset para detectar tipo
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

                # Detectar tipo de problema (clasificación vs regresión)
                y = df.iloc[:, -1]
                is_classification = y.dtype == 'object' or len(y.unique()) <= 10
                problem_type = 'classification' if is_classification else 'regression'

                # ============================================================
                # CREAR Y EJECUTAR PIPELINE POO ✨
                # ============================================================
                pipeline = MLPipeline(
                    algorithm=perfil.algoritmo_elegido,
                    problem_type=problem_type
                )

                # Obtener configuración de test_size del formulario
                test_size = int(request.POST.get('test_size', 30)) / 100

                # ============================================================
                # EJECUTAR PIPELINE COMPLETO CON POO
                # Esto reemplaza 150+ líneas de código inline ✅
                # ============================================================
                results = pipeline.run_complete_pipeline(
                    file_path=perfil.dataset_archivo.path,
                    test_size=test_size,
                    val_size=0.1,
                    use_cross_validation=True,      # ✅ Validación cruzada
                    hyperparameter_tuning=True,      # ✅ Ajuste de hiperparámetros
                    engineer_features=False
                )

                # ============================================================
                # EXTRAER MÉTRICAS DEL RESULTADO
                # ============================================================
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

                # ============================================================
                # GUARDAR MÉTRICAS EN EL PERFIL
                # ============================================================
                perfil.metricas = evaluation_metrics

                # ============================================================
                # GUARDAR PIPELINE COMPLETO
                # ============================================================
                modelo_path = os.path.join('media', 'modelos', f'pipeline_{perfil.id}.pkl')
                os.makedirs(os.path.dirname(modelo_path), exist_ok=True)

                # Guardar pipeline completo (incluye modelo + preprocesador)
                pipeline.save_pipeline(modelo_path)

                perfil.modelo_archivo = modelo_path.replace('media/', '')
                perfil.save()

                # ============================================================
                # MENSAJE DE ÉXITO
                # ============================================================
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

    # ========================================================================
    # RENDERIZAR TEMPLATE (igual que antes)
    # ========================================================================
    context = {
        'perfil': perfil,
        'algoritmos': PerfilEstudiante.ALGORITMOS_CHOICES
    }
    return render(request, 'dashboard/modelo.html', context)


# ============================================================================
# FUNCIÓN AUXILIAR: Migrar método en views.py
# ============================================================================
def como_integrar():
    """
    INSTRUCCIONES PARA INTEGRAR EN views.py ORIGINAL:

    OPCIÓN 1: Reemplazar completamente (RECOMENDADO)
    --------------------------------------------------
    1. Abrir apps/dashboard/views.py
    2. REEMPLAZAR todo el método dashboard_modelo (líneas 168-239)
       con el código de dashboard_modelo_poo de este archivo
    3. Renombrar dashboard_modelo_poo() a dashboard_modelo()
    4. ¡Listo! Ahora usa POO

    OPCIÓN 2: Mantener ambos (para testing)
    ----------------------------------------
    1. En apps/dashboard/urls.py, agregar una nueva ruta:
       path('<int:perfil_id>/modelo-poo/', views_poo.dashboard_modelo_poo, name='dashboard_modelo_poo')

    2. Importar en urls.py:
       from apps.dashboard import views_poo

    3. Ahora tienes dos versiones:
       - /dashboard/<id>/modelo/ → Versión anterior (inline)
       - /dashboard/<id>/modelo-poo/ → Nueva versión (POO)

    OPCIÓN 3: Gradual (más seguro)
    -------------------------------
    1. Primero probar con OPCIÓN 2
    2. Si funciona bien, aplicar OPCIÓN 1
    3. Eliminar este archivo views_poo.py
    """
    pass
