"""
Modulo ml_models: Sistema completo de Machine Learning con POO

Este modulo implementa un sistema completo de Machine Learning siguiendo
principios de Programacion Orientada a Objetos:

- Encapsulamiento
- Abstraccion
- Herencia
- Polimorfismo
- Single Responsibility Principle

Clases disponibles:
- DataLoader: Carga y gestion de datos
- DataPreprocessor: Preprocesamiento y transformacion
- FeatureEngineer: Ingenieria y seleccion de caracteristicas
- ModelTrainer: Entrenamiento y ajuste de modelos
- ModelEvaluator: Evaluacion y metricas
- Predictor: Predicciones en nuevos datos
- MLPipeline: Pipeline completo (clase orquestadora)

Ejemplo de uso basico:
    from apps.ml_models import MLPipeline

    # Crear pipeline
    pipeline = MLPipeline(algorithm='regresion_logistica', problem_type='classification')

    # Ejecutar pipeline completo
    results = pipeline.run_complete_pipeline(
        file_path='diabetes.csv',
        use_cross_validation=True,
        hyperparameter_tuning=True
    )

    # Ver metricas
    print(results['evaluation'])

Ejemplo de uso avanzado:
    from apps.ml_models import (
        DataLoader, DataPreprocessor, ModelTrainer,
        ModelEvaluator
    )

    # Carga personalizada
    loader = DataLoader()
    loader.load_csv('data.csv')
    X, y = loader.split_features_target()

    # Preprocesamiento personalizado
    preprocessor = DataPreprocessor(scaling_method='minmax')
    X_processed, y_processed = preprocessor.fit_transform(X, y)

    # Entrenamiento con ajuste de hiperparametros
    trainer = ModelTrainer('knn', 'classification')
    trainer.split_data(X_processed.values, y_processed.values)
    trainer.hyperparameter_tuning(X_processed.values, y_processed.values)

    # Evaluacion
    evaluator = ModelEvaluator('classification')
    y_pred = trainer.trainer.predict(trainer.X_test)
    metrics = evaluator.evaluate(trainer.y_test, y_pred)
"""

__version__ = '1.0.0'
__author__ = 'Dashboard ML - Aprendizaje Supervisado'

# Importar todas las clases principales
from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .model_trainer import (
    ModelTrainer,
    BaseModelTrainer,
    LinearRegressionTrainer,
    LogisticRegressionTrainer,
    RidgeLassoTrainer,
    DecisionTreeTrainer,
    KNNTrainer,
    NeuralNetworkTrainer
)
from .evaluator import ModelEvaluator
from .predictor import Predictor
from .pipeline import MLPipeline

# Definir que se exporta cuando se hace "from ml_models import *"
__all__ = [
    # Clases principales
    'DataLoader',
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'Predictor',
    'MLPipeline',

    # Trainers especificos (para uso avanzado)
    'BaseModelTrainer',
    'LinearRegressionTrainer',
    'LogisticRegressionTrainer',
    'RidgeLassoTrainer',
    'DecisionTreeTrainer',
    'KNNTrainer',
    'NeuralNetworkTrainer',
]
