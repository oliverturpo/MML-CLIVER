"""
Módulo Predictor: Realizar predicciones en nuevos datos
Implementa principios POO: Encapsulamiento y Single Responsibility Principle
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Clase responsable de realizar predicciones con modelos entrenados.

    Principios POO aplicados:
    - Encapsulamiento: Mantiene el modelo y transformadores
    - Single Responsibility: Solo se encarga de hacer predicciones

    Attributes:
        model: Modelo de ML entrenado
        preprocessor: Preprocesador para transformar nuevos datos
        feature_names: Nombres de las características esperadas
    """

    def __init__(self, model=None, preprocessor=None):
        """
        Inicializa el Predictor.

        Args:
            model: Modelo de ML entrenado
            preprocessor: DataPreprocessor para transformar datos
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names: Optional[List[str]] = None
        self.problem_type: Optional[str] = None

        if model is not None:
            logger.info("Predictor inicializado con modelo")
        else:
            logger.info("Predictor inicializado sin modelo")

    def load_model(self, model_path: str):
        """
        Carga un modelo desde disco.

        Args:
            model_path (str): Ruta del archivo del modelo
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # El modelo puede estar en diferentes formatos
            if isinstance(model_data, dict):
                # Formato completo con preprocesador
                self.model = model_data.get('model')
                self.preprocessor = model_data.get('preprocessor')
                self.feature_names = model_data.get('feature_names')
                self.problem_type = model_data.get('problem_type')
            else:
                # Solo el modelo
                self.model = model_data

            logger.info(f"Modelo cargado desde: {model_path}")

        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Guarda el modelo y configuración en disco.

        Args:
            model_path (str): Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'problem_type': self.problem_type
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Modelo guardado en: {model_path}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray],
               preprocess: bool = True) -> np.ndarray:
        """
        Realiza predicciones en nuevos datos.

        Args:
            X (pd.DataFrame o np.ndarray): Datos a predecir
            preprocess (bool): Si aplicar preprocesamiento

        Returns:
            np.ndarray: Predicciones
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Use load_model() o asigne un modelo")

        # Preprocesar si es necesario
        if preprocess and self.preprocessor is not None:
            if isinstance(X, pd.DataFrame):
                X_processed, _ = self.preprocessor.transform(X)
                X_processed = X_processed.values
            else:
                X_processed = X
        else:
            X_processed = X if isinstance(X, np.ndarray) else X.values

        # Realizar predicción
        predictions = self.model.predict(X_processed)

        logger.info(f"Predicciones realizadas para {len(predictions)} muestras")

        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray],
                     preprocess: bool = True) -> np.ndarray:
        """
        Retorna probabilidades de predicción (solo para clasificación).

        Args:
            X (pd.DataFrame o np.ndarray): Datos a predecir
            preprocess (bool): Si aplicar preprocesamiento

        Returns:
            np.ndarray: Probabilidades de predicción
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("El modelo no soporta predict_proba")

        # Preprocesar si es necesario
        if preprocess and self.preprocessor is not None:
            if isinstance(X, pd.DataFrame):
                X_processed, _ = self.preprocessor.transform(X)
                X_processed = X_processed.values
            else:
                X_processed = X
        else:
            X_processed = X if isinstance(X, np.ndarray) else X.values

        # Obtener probabilidades
        probabilities = self.model.predict_proba(X_processed)

        logger.info(f"Probabilidades calculadas para {len(probabilities)} muestras")

        return probabilities

    def predict_single(self, data: Dict) -> Union[float, int, str]:
        """
        Realiza predicción para una sola muestra desde un diccionario.

        Args:
            data (Dict): Diccionario con los valores de las características

        Returns:
            Union[float, int, str]: Predicción
        """
        # Convertir diccionario a DataFrame
        df = pd.DataFrame([data])

        # Asegurar que tenga las características correctas
        if self.feature_names is not None:
            # Verificar características faltantes
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Características faltantes: {missing_features}. "
                             f"Se rellenarán con valores por defecto.")
                for feat in missing_features:
                    df[feat] = 0

            # Ordenar columnas
            df = df[self.feature_names]

        # Predecir
        prediction = self.predict(df, preprocess=True)

        # Si el preprocesador tiene encoder de target, invertir transformación
        if (self.preprocessor is not None and
            hasattr(self.preprocessor, 'target_encoder') and
            self.preprocessor.target_encoder is not None):
            prediction = self.preprocessor.inverse_transform_target(prediction)

        return prediction[0]

    def predict_batch(self, data_list: List[Dict]) -> np.ndarray:
        """
        Realiza predicciones para un lote de muestras.

        Args:
            data_list (List[Dict]): Lista de diccionarios con datos

        Returns:
            np.ndarray: Array de predicciones
        """
        df = pd.DataFrame(data_list)

        # Asegurar características correctas
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Características faltantes: {missing_features}")
                for feat in missing_features:
                    df[feat] = 0
            df = df[self.feature_names]

        return self.predict(df, preprocess=True)

    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray],
                               threshold: float = 0.5) -> Dict:
        """
        Realiza predicción con información de confianza (solo clasificación).

        Args:
            X (pd.DataFrame o np.ndarray): Datos a predecir
            threshold (float): Umbral de confianza

        Returns:
            Dict: Predicciones, probabilidades y nivel de confianza
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("El modelo no soporta predict_proba")

        predictions = self.predict(X, preprocess=True)
        probabilities = self.predict_proba(X, preprocess=True)

        # Calcular confianza (máxima probabilidad)
        confidences = np.max(probabilities, axis=1)

        # Identificar predicciones de baja confianza
        low_confidence_mask = confidences < threshold

        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences,
            'low_confidence_indices': np.where(low_confidence_mask)[0].tolist(),
            'n_low_confidence': int(low_confidence_mask.sum())
        }

        logger.info(f"Predicciones con confianza: {results['n_low_confidence']} "
                   f"muestras con baja confianza (< {threshold})")

        return results

    def get_prediction_explanation(self, X: pd.DataFrame, index: int = 0) -> Dict:
        """
        Proporciona una explicación simple de la predicción.

        Args:
            X (pd.DataFrame): Datos
            index (int): Índice de la muestra a explicar

        Returns:
            Dict: Explicación de la predicción
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado")

        # Obtener predicción para la muestra
        sample = X.iloc[[index]]
        prediction = self.predict(sample, preprocess=True)[0]

        explanation = {
            'prediction': float(prediction),
            'input_values': sample.iloc[0].to_dict()
        }

        # Si el modelo tiene feature_importances o coef_, añadirlos
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = X.columns.tolist()
            explanation['feature_importance'] = {
                feat: float(imp) for feat, imp in zip(feature_names, importances)
            }
        elif hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            feature_names = X.columns.tolist()
            explanation['feature_coefficients'] = {
                feat: float(coef) for feat, coef in zip(feature_names, coefs)
            }

        return explanation

    def validate_input(self, X: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Valida que los datos de entrada sean correctos.

        Args:
            X (pd.DataFrame): Datos a validar

        Returns:
            Tuple[bool, Optional[str]]: (es_válido, mensaje_error)
        """
        # Verificar que sea DataFrame
        if not isinstance(X, pd.DataFrame):
            return False, "Los datos deben ser un DataFrame de pandas"

        # Verificar características si están definidas
        if self.feature_names is not None:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                return False, f"Faltan características: {missing}"

            extra = set(X.columns) - set(self.feature_names)
            if extra:
                logger.warning(f"Características extra que serán ignoradas: {extra}")

        # Verificar valores nulos
        null_counts = X.isnull().sum()
        if null_counts.any():
            logger.warning(f"Se encontraron valores nulos: {null_counts[null_counts > 0].to_dict()}")

        return True, None

    def set_feature_names(self, feature_names: List[str]):
        """
        Establece los nombres de las características esperadas.

        Args:
            feature_names (List[str]): Lista de nombres de características
        """
        self.feature_names = feature_names
        logger.info(f"Feature names establecidos: {len(feature_names)} características")

    def get_model_info(self) -> Dict:
        """
        Retorna información sobre el modelo cargado.

        Returns:
            Dict: Información del modelo
        """
        if self.model is None:
            return {"error": "No hay modelo cargado"}

        info = {
            "model_type": type(self.model).__name__,
            "has_preprocessor": self.preprocessor is not None,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "problem_type": self.problem_type
        }

        # Añadir parámetros del modelo si están disponibles
        if hasattr(self.model, 'get_params'):
            info["model_params"] = self.model.get_params()

        return info

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        model_type = type(self.model).__name__ if self.model else "None"
        return f"Predictor(model={model_type}, has_preprocessor={self.preprocessor is not None})"
