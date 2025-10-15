"""
Módulo DataPreprocessor: Preprocesamiento y transformación de datos
Implementa principios POO: Encapsulamiento, Abstracción y Single Responsibility Principle
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Clase responsable del preprocesamiento y transformación de datos.

    Principios POO aplicados:
    - Encapsulamiento: Mantiene el estado del preprocesamiento
    - Abstracción: Oculta la complejidad del preprocesamiento
    - Single Responsibility: Solo se encarga de preprocesar datos

    Attributes:
        scaler: Escalador de características (StandardScaler o MinMaxScaler)
        label_encoders: Diccionario de LabelEncoders para variables categóricas
        target_encoder: LabelEncoder para la variable objetivo
        imputer: Imputador para valores faltantes
        feature_names: Nombres de las características después del preprocesamiento
    """

    def __init__(self, scaling_method: str = 'standard'):
        """
        Inicializa el DataPreprocessor.

        Args:
            scaling_method (str): Método de escalado ('standard', 'minmax', 'none')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder: Optional[LabelEncoder] = None
        self.imputer_numeric: Optional[SimpleImputer] = None
        self.imputer_categorical: Optional[SimpleImputer] = None
        self.feature_names: Optional[List[str]] = None
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self._is_fitted = False

        # Inicializar escalador según el método
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Método de escalado '{scaling_method}' no válido")

        logger.info(f"DataPreprocessor inicializado con método de escalado: {scaling_method}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Ajusta los transformadores con los datos de entrenamiento.

        Args:
            X (pd.DataFrame): Características
            y (pd.Series, optional): Variable objetivo

        Returns:
            DataPreprocessor: self (para permitir method chaining)
        """
        logger.info("Iniciando ajuste del preprocesador...")

        # Identificar columnas categóricas y numéricas
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Columnas categóricas: {self.categorical_columns}")
        logger.info(f"Columnas numéricas: {self.numeric_columns}")

        # Ajustar LabelEncoders para columnas categóricas
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Manejar valores nulos
            valid_mask = X[col].notna()
            if valid_mask.any():
                le.fit(X[col][valid_mask].astype(str))
                self.label_encoders[col] = le

        # Ajustar imputadores
        if self.numeric_columns:
            self.imputer_numeric = SimpleImputer(strategy='median')
            self.imputer_numeric.fit(X[self.numeric_columns])

        if self.categorical_columns:
            self.imputer_categorical = SimpleImputer(strategy='constant', fill_value='missing')

        # Ajustar escalador si existe
        if self.scaler is not None and self.numeric_columns:
            # Primero imputar, luego escalar
            X_numeric_imputed = self.imputer_numeric.transform(X[self.numeric_columns])
            self.scaler.fit(X_numeric_imputed)

        # Ajustar encoder de target si es categórico
        if y is not None and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y.astype(str))
            logger.info(f"Target es categórico. Clases: {self.target_encoder.classes_}")

        self._is_fitted = True
        logger.info("Preprocesador ajustado exitosamente")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transforma los datos usando los transformadores ajustados.

        Args:
            X (pd.DataFrame): Características a transformar
            y (pd.Series, optional): Variable objetivo a transformar

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: (X_transformed, y_transformed)

        Raises:
            ValueError: Si el preprocesador no ha sido ajustado
        """
        if not self._is_fitted:
            raise ValueError("El preprocesador debe ser ajustado primero con fit()")

        logger.info("Transformando datos...")
        X_transformed = X.copy()

        # Transformar columnas categóricas
        for col in self.categorical_columns:
            if col in self.label_encoders:
                # Manejar valores nulos y nuevos valores
                X_transformed[col] = X_transformed[col].fillna('missing').astype(str)

                # Manejar valores no vistos durante fit
                le = self.label_encoders[col]
                def safe_transform(val):
                    if val in le.classes_:
                        return le.transform([val])[0]
                    else:
                        # Asignar el valor más frecuente o -1
                        return -1

                X_transformed[col] = X_transformed[col].apply(safe_transform)

        # Imputar valores faltantes en numéricas
        if self.numeric_columns and self.imputer_numeric:
            X_transformed[self.numeric_columns] = self.imputer_numeric.transform(
                X_transformed[self.numeric_columns]
            )

        # Escalar características numéricas
        if self.scaler is not None and self.numeric_columns:
            X_transformed[self.numeric_columns] = self.scaler.transform(
                X_transformed[self.numeric_columns]
            )

        # Transformar target si es necesario
        y_transformed = None
        if y is not None:
            if self.target_encoder is not None:
                y_transformed = pd.Series(
                    self.target_encoder.transform(y.astype(str)),
                    index=y.index
                )
                logger.info("Target transformado a valores numéricos")
            else:
                y_transformed = y.copy()

        self.feature_names = list(X_transformed.columns)
        logger.info(f"Transformación completada. Shape: {X_transformed.shape}")

        return X_transformed, y_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Ajusta y transforma los datos en un solo paso.

        Args:
            X (pd.DataFrame): Características
            y (pd.Series, optional): Variable objetivo

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: (X_transformed, y_transformed)
        """
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Invierte la transformación del target (útil para predicciones).

        Args:
            y (np.ndarray): Target transformado

        Returns:
            np.ndarray: Target en formato original
        """
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y.astype(int))
        return y

    def get_feature_names(self) -> List[str]:
        """
        Retorna los nombres de las características después del preprocesamiento.

        Returns:
            List[str]: Lista de nombres de características
        """
        if self.feature_names is None:
            raise ValueError("No hay características procesadas aún")
        return self.feature_names

    def get_categorical_mapping(self, column: str) -> Dict:
        """
        Obtiene el mapeo de una columna categórica.

        Args:
            column (str): Nombre de la columna

        Returns:
            Dict: Mapeo de valores originales a codificados
        """
        if column not in self.label_encoders:
            raise ValueError(f"Columna '{column}' no es categórica o no ha sido procesada")

        le = self.label_encoders[column]
        return {label: idx for idx, label in enumerate(le.classes_)}

    def handle_outliers(self, X: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Maneja valores atípicos en las características numéricas.

        Args:
            X (pd.DataFrame): Características
            method (str): Método para detectar outliers ('iqr', 'zscore')
            threshold (float): Umbral para considerar un valor como outlier

        Returns:
            pd.DataFrame: Datos sin outliers
        """
        X_clean = X.copy()

        for col in self.numeric_columns:
            if col not in X_clean.columns:
                continue

            if method == 'iqr':
                Q1 = X_clean[col].quantile(0.25)
                Q3 = X_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Reemplazar outliers con los límites
                X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)

            elif method == 'zscore':
                mean = X_clean[col].mean()
                std = X_clean[col].std()
                z_scores = np.abs((X_clean[col] - mean) / std)
                X_clean.loc[z_scores > threshold, col] = mean

        logger.info(f"Outliers manejados usando método: {method}")
        return X_clean

    def detect_problem_type(self, y: pd.Series) -> str:
        """
        Detecta automáticamente si es un problema de clasificación o regresión.

        Args:
            y (pd.Series): Variable objetivo

        Returns:
            str: 'classification' o 'regression'
        """
        # Si es tipo object, es clasificación
        if y.dtype == 'object':
            return 'classification'

        # Si tiene pocos valores únicos (menos de 10), probablemente es clasificación
        n_unique = len(y.unique())
        if n_unique <= 10:
            return 'classification'

        return 'regression'

    def get_preprocessing_info(self) -> Dict:
        """
        Retorna información sobre el preprocesamiento realizado.

        Returns:
            Dict: Información del preprocesamiento
        """
        info = {
            "is_fitted": self._is_fitted,
            "scaling_method": self.scaling_method,
            "n_categorical_columns": len(self.categorical_columns),
            "n_numeric_columns": len(self.numeric_columns),
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "has_target_encoder": self.target_encoder is not None
        }

        if self.target_encoder:
            info["target_classes"] = list(self.target_encoder.classes_)

        return info

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        status = "fitted" if self._is_fitted else "not fitted"
        return f"DataPreprocessor(scaling='{self.scaling_method}', status='{status}')"
