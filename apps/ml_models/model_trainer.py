"""
Módulo ModelTrainer: Entrenamiento y ajuste de modelos ML
Implementa principios POO: Encapsulamiento, Abstracción, Herencia y Polimorfismo
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Clase base abstracta para entrenadores de modelos.

    Principios POO aplicados:
    - Abstracción: Define interfaz común para todos los entrenadores
    - Herencia: Las clases específicas heredan de esta
    - Polimorfismo: Cada clase implementa su propia versión de train()
    """

    def __init__(self, algorithm_name: str):
        """
        Inicializa el entrenador base.

        Args:
            algorithm_name (str): Nombre del algoritmo
        """
        self.algorithm_name = algorithm_name
        self.model = None
        self.best_params: Optional[Dict] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Método abstracto para entrenar el modelo."""
        pass

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict:
        """Retorna hiperparámetros por defecto para GridSearch."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.

        Args:
            X (np.ndarray): Características

        Returns:
            np.ndarray: Predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.model.predict(X)

    def save_model(self, filepath: str):
        """
        Guarda el modelo entrenado en disco.

        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo desde disco.

        Args:
            filepath (str): Ruta del modelo guardado
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Modelo cargado desde: {filepath}")


class LinearRegressionTrainer(BaseModelTrainer):
    """Entrenador para Regresión Lineal."""

    def __init__(self):
        super().__init__("Regresión Lineal")
        self.model = LinearRegression()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Entrena el modelo de regresión lineal."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Regresión lineal no tiene hiperparámetros a ajustar."""
        return {'fit_intercept': [True, False]}

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Retorna los coeficientes del modelo."""
        if self.is_trained:
            return self.model.coef_
        return None


class LogisticRegressionTrainer(BaseModelTrainer):
    """Entrenador para Regresión Logística."""

    def __init__(self):
        super().__init__("Regresión Logística")
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Entrena el modelo de regresión logística."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros para GridSearch."""
        return {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades de predicción."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.model.predict_proba(X)


class RidgeLassoTrainer(BaseModelTrainer):
    """Entrenador para Ridge y Lasso."""

    def __init__(self, model_type: str = 'ridge'):
        super().__init__(f"Regresión {model_type.capitalize()}")
        if model_type.lower() == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type.lower() == 'lasso':
            self.model = Lasso(alpha=1.0, random_state=42)
        else:
            raise ValueError("model_type debe ser 'ridge' o 'lasso'")
        self.model_type = model_type

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entrena el modelo Ridge o Lasso."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros para GridSearch."""
        return {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }


class DecisionTreeTrainer(BaseModelTrainer):
    """Entrenador para Árboles de Decisión."""

    def __init__(self, problem_type: str = 'regression'):
        super().__init__("Árbol de Decisión CART")
        self.problem_type = problem_type
        if problem_type == 'regression':
            self.model = DecisionTreeRegressor(random_state=42)
        else:
            self.model = DecisionTreeClassifier(random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entrena el árbol de decisión."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros para GridSearch."""
        return {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Retorna la importancia de las características."""
        if self.is_trained:
            return self.model.feature_importances_
        return None


class KNNTrainer(BaseModelTrainer):
    """Entrenador para K-Nearest Neighbors."""

    def __init__(self, problem_type: str = 'regression'):
        super().__init__("K-Nearest Neighbors")
        self.problem_type = problem_type
        if problem_type == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=5)
        else:
            self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entrena el modelo KNN."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros para GridSearch."""
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }


class NeuralNetworkTrainer(BaseModelTrainer):
    """Entrenador para Redes Neuronales."""

    def __init__(self, problem_type: str = 'regression'):
        super().__init__("Red Neuronal")
        self.problem_type = problem_type
        if problem_type == 'regression':
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50),
                                     max_iter=1000, random_state=42)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50),
                                      max_iter=1000, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entrena la red neuronal."""
        logger.info(f"Entrenando {self.algorithm_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Entrenamiento completado")
        return self.model

    def get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros para GridSearch."""
        return {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }


class ModelTrainer:
    """
    Clase principal para gestionar el entrenamiento de modelos.

    Esta clase implementa:
    - Validación cruzada
    - Ajuste de hiperparámetros (GridSearch/RandomizedSearch)
    - División train/validation/test
    """

    def __init__(self, algorithm: str, problem_type: str = 'regression'):
        """
        Inicializa el ModelTrainer.

        Args:
            algorithm (str): Nombre del algoritmo
            problem_type (str): 'regression' o 'classification'
        """
        self.algorithm = algorithm
        self.problem_type = problem_type
        self.trainer = self._get_trainer()

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        logger.info(f"ModelTrainer inicializado: {algorithm} ({problem_type})")

    def _get_trainer(self) -> BaseModelTrainer:
        """Retorna el entrenador apropiado según el algoritmo."""
        trainers = {
            'regresion_lineal': LinearRegressionTrainer(),
            'regresion_logistica': LogisticRegressionTrainer(),
            'ridge': RidgeLassoTrainer('ridge'),
            'lasso': RidgeLassoTrainer('lasso'),
            'ridge_lasso': RidgeLassoTrainer('ridge'),  # Por defecto Ridge
            'arbol_cart': DecisionTreeTrainer(self.problem_type),
            'knn': KNNTrainer(self.problem_type),
            'red_neuronal': NeuralNetworkTrainer(self.problem_type)
        }

        if self.algorithm not in trainers:
            raise ValueError(f"Algoritmo '{self.algorithm}' no soportado")

        return trainers[self.algorithm]

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Tuple:
        """
        Divide los datos en train, validation y test sets.

        Args:
            X (np.ndarray): Características
            y (np.ndarray): Variable objetivo
            test_size (float): Proporción para test
            val_size (float): Proporción para validación (del conjunto de entrenamiento)
            random_state (int): Semilla aleatoria

        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Primero separar test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Luego separar train y validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        logger.info(f"Datos divididos - Train: {X_train.shape}, "
                   f"Val: {X_val.shape if X_val is not None else 'None'}, "
                   f"Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train: Optional[np.ndarray] = None,
              y_train: Optional[np.ndarray] = None):
        """
        Entrena el modelo.

        Args:
            X_train (np.ndarray, optional): Características de entrenamiento
            y_train (np.ndarray, optional): Variable objetivo de entrenamiento
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("Debe proporcionar datos de entrenamiento")

        self.trainer.train(X_train, y_train)

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: Optional[str] = None) -> Dict:
        """
        Realiza validación cruzada.

        Args:
            X (np.ndarray): Características
            y (np.ndarray): Variable objetivo
            cv (int): Número de folds
            scoring (str, optional): Métrica a usar

        Returns:
            Dict: Resultados de la validación cruzada
        """
        logger.info(f"Realizando validación cruzada con {cv} folds...")

        if scoring is None:
            scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'

        scores = cross_val_score(self.trainer.model, X, y, cv=cv, scoring=scoring)
        self.trainer.cv_scores = scores

        results = {
            'scores': scores.tolist(),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'scoring_metric': scoring
        }

        logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        return results

    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray,
                             param_grid: Optional[Dict] = None,
                             cv: int = 5, method: str = 'grid') -> Dict:
        """
        Ajusta hiperparámetros usando GridSearch o RandomizedSearch.

        Args:
            X (np.ndarray): Características
            y (np.ndarray): Variable objetivo
            param_grid (Dict, optional): Grid de parámetros. Si es None, usa valores por defecto
            cv (int): Número de folds para cross-validation
            method (str): 'grid' o 'random'

        Returns:
            Dict: Mejores parámetros y scores
        """
        if param_grid is None:
            param_grid = self.trainer.get_default_hyperparameters()

        logger.info(f"Iniciando ajuste de hiperparámetros ({method})...")
        logger.info(f"Parámetros a probar: {param_grid}")

        scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'

        if method == 'grid':
            search = GridSearchCV(
                self.trainer.model, param_grid, cv=cv,
                scoring=scoring, n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                self.trainer.model, param_grid, cv=cv,
                scoring=scoring, n_jobs=-1, n_iter=20, verbose=1
            )

        search.fit(X, y)

        self.trainer.model = search.best_estimator_
        self.trainer.best_params = search.best_params_
        self.trainer.is_trained = True

        results = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results': search.cv_results_
        }

        logger.info(f"Mejores parámetros: {results['best_params']}")
        logger.info(f"Mejor score: {results['best_score']:.4f}")

        return results

    def get_model(self):
        """Retorna el modelo entrenado."""
        return self.trainer.model

    def get_trainer(self) -> BaseModelTrainer:
        """Retorna el objeto trainer."""
        return self.trainer

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        return f"ModelTrainer(algorithm='{self.algorithm}', type='{self.problem_type}')"
