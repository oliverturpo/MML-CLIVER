"""
Módulo MLPipeline: Pipeline completo de Machine Learning
Implementa principios POO: Encapsulamiento, Composición y Orchestration Pattern
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
import pickle
import json
from pathlib import Path
import logging

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Pipeline completo de Machine Learning que orquesta todas las clases.

    Principios POO aplicados:
    - Encapsulamiento: Encapsula todo el flujo de ML
    - Composición: Usa instancias de otras clases
    - Orchestration: Coordina el flujo completo del proceso

    Esta clase implementa el patrón Facade, proporcionando una interfaz
    simple para todo el proceso de ML.

    Attributes:
        data_loader: Instancia de DataLoader
        preprocessor: Instancia de DataPreprocessor
        feature_engineer: Instancia de FeatureEngineer
        trainer: Instancia de ModelTrainer
        evaluator: Instancia de ModelEvaluator
        predictor: Instancia de Predictor
    """

    def __init__(self, algorithm: str, problem_type: str = 'auto'):
        """
        Inicializa el MLPipeline.

        Args:
            algorithm (str): Algoritmo a usar
            problem_type (str): 'classification', 'regression' o 'auto'
        """
        self.algorithm = algorithm
        self.problem_type = problem_type

        # Inicializar componentes
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor(scaling_method='standard')
        self.feature_engineer = FeatureEngineer()
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.predictor = Predictor()

        # Estado del pipeline
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.is_fitted = False
        self.metrics: Dict = {}
        self.pipeline_config: Dict = {}

        logger.info(f"MLPipeline inicializado: {algorithm} ({problem_type})")

    def load_data(self, file_path: str = None, dataframe: pd.DataFrame = None,
                  target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Carga datos desde un archivo o DataFrame.

        Args:
            file_path (str, optional): Ruta del archivo CSV
            dataframe (pd.DataFrame, optional): DataFrame ya cargado
            target_column (str, optional): Nombre de la columna objetivo

        Returns:
            pd.DataFrame: Datos cargados
        """
        logger.info("=== PASO 1: CARGA DE DATOS ===")

        if file_path:
            self.data_loader.load_csv(file_path)
        elif dataframe is not None:
            self.data_loader.load_from_dataframe(dataframe)
        else:
            raise ValueError("Debe proporcionar file_path o dataframe")

        # Separar X e y
        X, y = self.data_loader.split_features_target(target_column)

        # Detectar tipo de problema si es 'auto'
        if self.problem_type == 'auto':
            self.problem_type = self.preprocessor.detect_problem_type(y)
            logger.info(f"Tipo de problema detectado: {self.problem_type}")

        logger.info(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
        return self.data_loader.get_data()

    def preprocess_data(self, X: Optional[pd.DataFrame] = None,
                       y: Optional[pd.Series] = None,
                       handle_outliers: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesa los datos.

        Args:
            X (pd.DataFrame, optional): Características
            y (pd.Series, optional): Variable objetivo
            handle_outliers (bool): Si manejar valores atípicos

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_processed, y_processed)
        """
        logger.info("=== PASO 2: PREPROCESAMIENTO ===")

        if X is None or y is None:
            X, y = self.data_loader.split_features_target()

        # Manejar outliers si se solicita
        if handle_outliers:
            X = self.preprocessor.handle_outliers(X, method='iqr')

        # Ajustar y transformar
        X_processed, y_processed = self.preprocessor.fit_transform(X, y)

        logger.info("Preprocesamiento completado")
        return X_processed, y_processed

    def engineer_features(self, X: pd.DataFrame,
                         create_polynomial: bool = False,
                         create_interactions: bool = False,
                         select_k_best: Optional[int] = None) -> pd.DataFrame:
        """
        Realiza ingeniería de características.

        Args:
            X (pd.DataFrame): Características preprocesadas
            create_polynomial (bool): Crear características polinómicas
            create_interactions (bool): Crear características de interacción
            select_k_best (int, optional): Número de mejores características a seleccionar

        Returns:
            pd.DataFrame: Características transformadas
        """
        logger.info("=== PASO 3: INGENIERÍA DE CARACTERÍSTICAS ===")

        X_engineered = X.copy()

        # Crear características polinómicas
        if create_polynomial:
            X_engineered = self.feature_engineer.create_polynomial_features(
                X_engineered, degree=2
            )

        # Crear características de interacción
        if create_interactions:
            X_engineered = self.feature_engineer.create_interaction_features(
                X_engineered
            )

        # Seleccionar mejores características
        if select_k_best and select_k_best < X_engineered.shape[1]:
            y = self.preprocessor.transform(X, self.y_train)[1]
            self.feature_engineer.select_k_best_features(
                X_engineered, y, k=select_k_best,
                problem_type=self.problem_type
            )
            X_engineered = self.feature_engineer.transform_selected_features(X_engineered)

        logger.info(f"Ingeniería completada: {X_engineered.shape[1]} características")
        return X_engineered

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42):
        """
        Divide los datos en train, validation y test.

        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            test_size (float): Proporción para test
            val_size (float): Proporción para validation
            random_state (int): Semilla aleatoria
        """
        logger.info("=== PASO 4: DIVISIÓN DE DATOS ===")

        # Inicializar trainer si no existe
        if self.trainer is None:
            self.trainer = ModelTrainer(self.algorithm, self.problem_type)

        # Dividir datos
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.trainer.split_data(X.values, y.values, test_size, val_size, random_state)

        logger.info(f"Train: {self.X_train.shape}, Val: {self.X_val.shape if self.X_val is not None else 'None'}, "
                   f"Test: {self.X_test.shape}")

    def train_model(self, use_cross_validation: bool = False,
                   cv_folds: int = 5,
                   hyperparameter_tuning: bool = False,
                   tuning_method: str = 'grid') -> Dict:
        """
        Entrena el modelo.

        Args:
            use_cross_validation (bool): Usar validación cruzada
            cv_folds (int): Número de folds para CV
            hyperparameter_tuning (bool): Ajustar hiperparámetros
            tuning_method (str): 'grid' o 'random'

        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("=== PASO 5: ENTRENAMIENTO DEL MODELO ===")

        if self.trainer is None:
            raise ValueError("Debe dividir los datos primero con split_data()")

        results = {}

        # Validación cruzada
        if use_cross_validation:
            logger.info(f"Ejecutando validación cruzada ({cv_folds} folds)...")
            cv_results = self.trainer.cross_validate(
                self.X_train, self.y_train, cv=cv_folds
            )
            results['cross_validation'] = cv_results

        # Ajuste de hiperparámetros
        if hyperparameter_tuning:
            logger.info(f"Ajustando hiperparámetros ({tuning_method} search)...")
            tuning_results = self.trainer.hyperparameter_tuning(
                self.X_train, self.y_train, method=tuning_method, cv=cv_folds
            )
            results['hyperparameter_tuning'] = tuning_results
        else:
            # Entrenamiento simple
            self.trainer.train(self.X_train, self.y_train)

        self.is_fitted = True
        logger.info("Entrenamiento completado exitosamente")

        return results

    def evaluate_model(self, on_validation: bool = False) -> Dict:
        """
        Evalúa el modelo.

        Args:
            on_validation (bool): Evaluar en validation set (si no, usa test set)

        Returns:
            Dict: Métricas de evaluación
        """
        logger.info("=== PASO 6: EVALUACIÓN DEL MODELO ===")

        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        # Seleccionar conjunto de datos
        if on_validation and self.X_val is not None:
            X_eval = self.X_val
            y_eval = self.y_val
            eval_type = "validation"
        else:
            X_eval = self.X_test
            y_eval = self.y_test
            eval_type = "test"

        # Inicializar evaluador
        self.evaluator = ModelEvaluator(self.problem_type)

        # Hacer predicciones
        y_pred = self.trainer.get_trainer().predict(X_eval)

        # Obtener probabilidades si es clasificación
        y_pred_proba = None
        if self.problem_type == 'classification':
            trainer_obj = self.trainer.get_trainer()
            if hasattr(trainer_obj, 'predict_proba'):
                y_pred_proba = trainer_obj.predict_proba(X_eval)

        # Evaluar
        self.metrics = self.evaluator.evaluate(y_eval, y_pred, y_pred_proba)
        self.metrics['eval_type'] = eval_type

        # Imprimir resumen
        print(self.evaluator.get_summary())

        logger.info(f"Evaluación completada en conjunto {eval_type}")
        return self.metrics

    def run_complete_pipeline(self, file_path: str = None,
                             dataframe: pd.DataFrame = None,
                             target_column: Optional[str] = None,
                             test_size: float = 0.2,
                             val_size: float = 0.1,
                             use_cross_validation: bool = True,
                             hyperparameter_tuning: bool = True,
                             engineer_features: bool = False) -> Dict:
        """
        Ejecuta el pipeline completo de principio a fin.

        Args:
            file_path (str, optional): Ruta del archivo
            dataframe (pd.DataFrame, optional): DataFrame
            target_column (str, optional): Columna objetivo
            test_size (float): Proporción de test
            val_size (float): Proporción de validation
            use_cross_validation (bool): Usar CV
            hyperparameter_tuning (bool): Ajustar hiperparámetros
            engineer_features (bool): Hacer ingeniería de características

        Returns:
            Dict: Resultados completos del pipeline
        """
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING")
        logger.info("="*60)

        results = {
            'success': False,
            'error': None
        }

        try:
            # 1. Cargar datos
            self.load_data(file_path, dataframe, target_column)
            X, y = self.data_loader.split_features_target(target_column)

            # 2. Preprocesar
            X_processed, y_processed = self.preprocess_data(X, y, handle_outliers=True)

            # 3. Ingeniería de características (opcional)
            if engineer_features:
                X_processed = self.engineer_features(X_processed, create_interactions=True)

            # 4. Dividir datos
            self.split_data(X_processed, y_processed, test_size, val_size)

            # 5. Entrenar
            training_results = self.train_model(
                use_cross_validation=use_cross_validation,
                hyperparameter_tuning=hyperparameter_tuning
            )
            results['training'] = training_results

            # 6. Evaluar
            eval_results = self.evaluate_model(on_validation=False)
            results['evaluation'] = eval_results

            # 7. Configurar predictor
            self.predictor.model = self.trainer.get_model()
            self.predictor.preprocessor = self.preprocessor
            self.predictor.feature_names = list(X_processed.columns)
            self.predictor.problem_type = self.problem_type

            results['success'] = True
            results['pipeline_config'] = {
                'algorithm': self.algorithm,
                'problem_type': self.problem_type,
                'n_features': X_processed.shape[1],
                'n_samples': X_processed.shape[0]
            }

            logger.info("="*60)
            logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Error en el pipeline: {str(e)}")
            results['error'] = str(e)
            raise

        return results

    def save_pipeline(self, filepath: str):
        """
        Guarda el pipeline completo en disco.

        Args:
            filepath (str): Ruta donde guardar el pipeline
        """
        if not self.is_fitted:
            raise ValueError("El pipeline debe ser entrenado primero")

        pipeline_data = {
            'model': self.trainer.get_model(),
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'algorithm': self.algorithm,
            'problem_type': self.problem_type,
            'metrics': self.metrics,
            'pipeline_config': self.pipeline_config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)

        logger.info(f"Pipeline guardado en: {filepath}")

    def load_pipeline(self, filepath: str):
        """
        Carga un pipeline completo desde disco.

        Args:
            filepath (str): Ruta del archivo del pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)

        self.trainer = ModelTrainer(pipeline_data['algorithm'], pipeline_data['problem_type'])
        self.trainer.trainer.model = pipeline_data['model']
        self.trainer.trainer.is_trained = True

        self.preprocessor = pipeline_data['preprocessor']
        self.feature_engineer = pipeline_data.get('feature_engineer', FeatureEngineer())
        self.algorithm = pipeline_data['algorithm']
        self.problem_type = pipeline_data['problem_type']
        self.metrics = pipeline_data.get('metrics', {})
        self.pipeline_config = pipeline_data.get('pipeline_config', {})

        # Configurar predictor
        self.predictor.model = pipeline_data['model']
        self.predictor.preprocessor = self.preprocessor
        self.predictor.problem_type = self.problem_type

        self.is_fitted = True
        logger.info(f"Pipeline cargado desde: {filepath}")

    def predict_new_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Hace predicciones en nuevos datos.

        Args:
            data (pd.DataFrame): Nuevos datos

        Returns:
            np.ndarray: Predicciones
        """
        if not self.is_fitted:
            raise ValueError("El pipeline debe ser entrenado primero")

        return self.predictor.predict(data, preprocess=True)

    def get_feature_importance(self) -> Optional[Dict]:
        """
        Obtiene la importancia de las características.

        Returns:
            Dict: Importancia de características si está disponible
        """
        if not self.is_fitted:
            return None

        trainer_obj = self.trainer.get_trainer()

        # Feature importances (árboles)
        if hasattr(trainer_obj, 'get_feature_importances'):
            importances = trainer_obj.get_feature_importances()
            if importances is not None:
                feature_names = self.preprocessor.get_feature_names()
                return {
                    'features': feature_names,
                    'importances': importances.tolist()
                }

        # Coeficientes (modelos lineales)
        if hasattr(trainer_obj, 'get_coefficients'):
            coefs = trainer_obj.get_coefficients()
            if coefs is not None:
                feature_names = self.preprocessor.get_feature_names()
                return {
                    'features': feature_names,
                    'coefficients': coefs.tolist()
                }

        return None

    def get_pipeline_summary(self) -> Dict:
        """
        Retorna un resumen completo del pipeline.

        Returns:
            Dict: Resumen del pipeline
        """
        summary = {
            'algorithm': self.algorithm,
            'problem_type': self.problem_type,
            'is_fitted': self.is_fitted,
            'preprocessing': self.preprocessor.get_preprocessing_info() if self.is_fitted else None,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance()
        }

        return summary

    def export_report(self, filepath: str, format: str = 'json'):
        """
        Exporta un reporte completo del pipeline.

        Args:
            filepath (str): Ruta donde guardar el reporte
            format (str): Formato ('json' o 'txt')
        """
        summary = self.get_pipeline_summary()

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
        elif format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("REPORTE DE MACHINE LEARNING\n")
                f.write("="*60 + "\n\n")
                f.write(f"Algoritmo: {self.algorithm}\n")
                f.write(f"Tipo de problema: {self.problem_type}\n")
                f.write(f"Estado: {'Entrenado' if self.is_fitted else 'No entrenado'}\n\n")

                if self.metrics:
                    f.write("MÉTRICAS:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in self.metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.4f}\n")

        logger.info(f"Reporte exportado a: {filepath}")

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"MLPipeline(algorithm='{self.algorithm}', type='{self.problem_type}', status='{status}')"
