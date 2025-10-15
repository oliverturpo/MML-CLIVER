"""
Módulo FeatureEngineer: Creación y selección de características
Implementa principios POO: Encapsulamiento, Abstracción y Single Responsibility Principle
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Clase responsable de la ingeniería y selección de características.

    Principios POO aplicados:
    - Encapsulamiento: Mantiene el estado de las transformaciones
    - Abstracción: Oculta la complejidad de la ingeniería de características
    - Single Responsibility: Solo se encarga de crear/seleccionar características

    Attributes:
        selected_features: Lista de características seleccionadas
        selector: Objeto de selección de características
        pca: Objeto PCA si se usa reducción de dimensionalidad
        feature_importances: Importancia de cada característica
    """

    def __init__(self):
        """Inicializa el FeatureEngineer."""
        self.selected_features: Optional[List[str]] = None
        self.selector = None
        self.pca: Optional[PCA] = None
        self.feature_importances: Optional[Dict[str, float]] = None
        self._is_fitted = False

        logger.info("FeatureEngineer inicializado")

    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2,
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea características polinómicas de las columnas especificadas.

        Args:
            X (pd.DataFrame): Características originales
            degree (int): Grado del polinomio
            columns (List[str], optional): Columnas a transformar. Si es None, usa todas las numéricas

        Returns:
            pd.DataFrame: DataFrame con características polinómicas añadidas
        """
        X_poly = X.copy()

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in X.columns:
                for d in range(2, degree + 1):
                    X_poly[f'{col}_pow{d}'] = X[col] ** d
                    logger.info(f"Creada característica: {col}_pow{d}")

        logger.info(f"Características polinómicas creadas. Shape: {X_poly.shape}")
        return X_poly

    def create_interaction_features(self, X: pd.DataFrame,
                                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea características de interacción (multiplicación) entre pares de columnas.

        Args:
            X (pd.DataFrame): Características originales
            columns (List[str], optional): Columnas a usar. Si es None, usa todas las numéricas

        Returns:
            pd.DataFrame: DataFrame con características de interacción añadidas
        """
        X_interact = X.copy()

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # Crear interacciones entre pares de columnas
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if col1 in X.columns and col2 in X.columns:
                    interaction_name = f'{col1}_x_{col2}'
                    X_interact[interaction_name] = X[col1] * X[col2]
                    logger.info(f"Creada interacción: {interaction_name}")

        logger.info(f"Características de interacción creadas. Shape: {X_interact.shape}")
        return X_interact

    def create_binning_features(self, X: pd.DataFrame, column: str,
                                n_bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
        """
        Crea características binned (agrupadas en intervalos).

        Args:
            X (pd.DataFrame): Características originales
            column (str): Columna a discretizar
            n_bins (int): Número de bins
            strategy (str): Estrategia de binning ('quantile', 'uniform', 'kmeans')

        Returns:
            pd.DataFrame: DataFrame con característica binned añadida
        """
        X_binned = X.copy()

        if column not in X.columns:
            raise ValueError(f"Columna '{column}' no existe en el DataFrame")

        if strategy == 'quantile':
            X_binned[f'{column}_binned'] = pd.qcut(X[column], q=n_bins, labels=False, duplicates='drop')
        elif strategy == 'uniform':
            X_binned[f'{column}_binned'] = pd.cut(X[column], bins=n_bins, labels=False)
        else:
            raise ValueError(f"Estrategia '{strategy}' no válida")

        logger.info(f"Característica binned creada: {column}_binned")
        return X_binned

    def select_k_best_features(self, X: pd.DataFrame, y: pd.Series,
                               k: int = 10, problem_type: str = 'classification') -> 'FeatureEngineer':
        """
        Selecciona las k mejores características usando pruebas estadísticas.

        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            k (int): Número de características a seleccionar
            problem_type (str): 'classification' o 'regression'

        Returns:
            FeatureEngineer: self (para method chaining)
        """
        # Seleccionar función de score apropiada
        if problem_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression

        # Ajustar selector
        self.selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        self.selector.fit(X, y)

        # Obtener características seleccionadas
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()

        # Calcular importancias
        scores = self.selector.scores_
        self.feature_importances = {
            col: score for col, score in zip(X.columns, scores)
        }

        self._is_fitted = True
        logger.info(f"Seleccionadas {len(self.selected_features)} características de {X.shape[1]}")
        logger.info(f"Features seleccionadas: {self.selected_features}")

        return self

    def transform_selected_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma el DataFrame para incluir solo las características seleccionadas.

        Args:
            X (pd.DataFrame): Características originales

        Returns:
            pd.DataFrame: DataFrame con solo las características seleccionadas

        Raises:
            ValueError: Si no se han seleccionado características
        """
        if not self._is_fitted or self.selected_features is None:
            raise ValueError("Debe ejecutar select_k_best_features() primero")

        return X[self.selected_features]

    def apply_pca(self, X: pd.DataFrame, n_components: int = None,
                  variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Aplica PCA para reducción de dimensionalidad.

        Args:
            X (pd.DataFrame): Características originales
            n_components (int, optional): Número de componentes. Si es None, usa variance_threshold
            variance_threshold (float): Varianza explicada mínima deseada

        Returns:
            pd.DataFrame: DataFrame con componentes principales
        """
        if n_components is None:
            self.pca = PCA(n_components=variance_threshold, svd_solver='full')
        else:
            self.pca = PCA(n_components=n_components)

        X_pca = self.pca.fit_transform(X)

        # Crear DataFrame con componentes principales
        columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=columns, index=X.index)

        logger.info(f"PCA aplicado. Componentes: {X_pca.shape[1]}, "
                   f"Varianza explicada: {self.pca.explained_variance_ratio_.sum():.2%}")

        return X_pca_df

    def get_correlation_with_target(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Calcula la correlación de cada característica con el target.

        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo

        Returns:
            pd.Series: Correlaciones ordenadas de mayor a menor (valor absoluto)
        """
        # Solo para características numéricas
        X_numeric = X.select_dtypes(include=[np.number])

        correlations = {}
        for col in X_numeric.columns:
            corr = np.corrcoef(X_numeric[col], y)[0, 1]
            correlations[col] = corr

        corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
        logger.info(f"Correlaciones calculadas para {len(corr_series)} características")

        return corr_series

    def remove_correlated_features(self, X: pd.DataFrame,
                                   threshold: float = 0.95) -> pd.DataFrame:
        """
        Elimina características altamente correlacionadas entre sí.

        Args:
            X (pd.DataFrame): Características
            threshold (float): Umbral de correlación para eliminar

        Returns:
            pd.DataFrame: DataFrame sin características correlacionadas
        """
        X_numeric = X.select_dtypes(include=[np.number])

        # Calcular matriz de correlación
        corr_matrix = X_numeric.corr().abs()

        # Seleccionar el triángulo superior de la matriz
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Encontrar características a eliminar
        to_drop = [column for column in upper_tri.columns
                   if any(upper_tri[column] > threshold)]

        logger.info(f"Eliminadas {len(to_drop)} características correlacionadas: {to_drop}")

        return X.drop(columns=to_drop)

    def create_aggregation_features(self, X: pd.DataFrame,
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea características de agregación (suma, promedio, etc.).

        Args:
            X (pd.DataFrame): Características originales
            columns (List[str], optional): Columnas a agregar. Si es None, usa todas las numéricas

        Returns:
            pd.DataFrame: DataFrame con características de agregación añadidas
        """
        X_agg = X.copy()

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) > 1:
            X_agg['feature_sum'] = X[columns].sum(axis=1)
            X_agg['feature_mean'] = X[columns].mean(axis=1)
            X_agg['feature_std'] = X[columns].std(axis=1)
            X_agg['feature_max'] = X[columns].max(axis=1)
            X_agg['feature_min'] = X[columns].min(axis=1)

            logger.info("Características de agregación creadas")

        return X_agg

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Retorna la importancia de las características.

        Returns:
            Dict[str, float]: Diccionario con importancia de características
        """
        return self.feature_importances

    def get_selected_features(self) -> Optional[List[str]]:
        """
        Retorna la lista de características seleccionadas.

        Returns:
            List[str]: Lista de características seleccionadas
        """
        return self.selected_features

    def get_pca_info(self) -> Optional[Dict]:
        """
        Retorna información sobre el PCA aplicado.

        Returns:
            Dict: Información del PCA
        """
        if self.pca is None:
            return None

        return {
            "n_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(self.pca.explained_variance_ratio_).tolist()
        }

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        n_selected = len(self.selected_features) if self.selected_features else 0
        return f"FeatureEngineer(selected_features={n_selected}, fitted={self._is_fitted})"
