"""
Módulo DataLoader: Carga y gestión de datos
Implementa principios POO: Encapsulamiento y Single Responsibility Principle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Clase responsable de la carga y gestión de datos desde diferentes fuentes.

    Principios POO aplicados:
    - Encapsulamiento: Los datos y métodos están encapsulados en la clase
    - Single Responsibility: Solo se encarga de cargar datos

    Attributes:
        file_path (str): Ruta del archivo de datos
        data (pd.DataFrame): DataFrame con los datos cargados
        encodings (list): Lista de encodings a intentar para leer CSV
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Inicializa el DataLoader.

        Args:
            file_path (str, optional): Ruta del archivo a cargar
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self.encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        self._original_data: Optional[pd.DataFrame] = None

        logger.info(f"DataLoader inicializado con archivo: {file_path}")

    def load_csv(self, file_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga un archivo CSV con manejo robusto de encodings.

        Args:
            file_path (str, optional): Ruta del archivo CSV
            **kwargs: Argumentos adicionales para pd.read_csv

        Returns:
            pd.DataFrame: DataFrame con los datos cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si no se puede leer con ningún encoding
        """
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            raise ValueError("Debe proporcionar una ruta de archivo")

        # Verificar que el archivo existe
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"El archivo {self.file_path} no existe")

        # Intentar leer con diferentes encodings
        for encoding in self.encodings:
            try:
                self.data = pd.read_csv(self.file_path, encoding=encoding, **kwargs)
                self._original_data = self.data.copy()
                logger.info(f"Archivo cargado exitosamente con encoding: {encoding}")
                logger.info(f"Dimensiones: {self.data.shape}")
                return self.data
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error con encoding {encoding}: {str(e)}")
                continue

        raise ValueError(f"No se pudo leer el archivo con ninguno de los encodings: {self.encodings}")

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Carga datos desde un DataFrame existente.

        Args:
            df (pd.DataFrame): DataFrame a cargar

        Returns:
            pd.DataFrame: DataFrame cargado
        """
        self.data = df.copy()
        self._original_data = df.copy()
        logger.info(f"DataFrame cargado. Dimensiones: {self.data.shape}")
        return self.data

    def get_data(self) -> pd.DataFrame:
        """
        Retorna los datos cargados.

        Returns:
            pd.DataFrame: Datos cargados

        Raises:
            ValueError: Si no se han cargado datos
        """
        if self.data is None:
            raise ValueError("No se han cargado datos. Use load_csv() o load_from_dataframe()")
        return self.data

    def get_original_data(self) -> pd.DataFrame:
        """
        Retorna una copia de los datos originales sin modificar.

        Returns:
            pd.DataFrame: Copia de los datos originales
        """
        if self._original_data is None:
            raise ValueError("No se han cargado datos originales")
        return self._original_data.copy()

    def get_info(self) -> dict:
        """
        Obtiene información básica sobre el dataset cargado.

        Returns:
            dict: Diccionario con información del dataset
        """
        if self.data is None:
            return {"error": "No hay datos cargados"}

        info = {
            "n_rows": len(self.data),
            "n_columns": len(self.data.columns),
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2
        }

        return info

    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Obtiene una muestra de los datos.

        Args:
            n (int): Número de filas a retornar

        Returns:
            pd.DataFrame: Muestra de los datos
        """
        if self.data is None:
            raise ValueError("No se han cargado datos")
        return self.data.head(n)

    def split_features_target(self, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa las características (X) de la variable objetivo (y).

        Args:
            target_column (str, optional): Nombre de la columna objetivo.
                                          Si es None, usa la última columna.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        if self.data is None:
            raise ValueError("No se han cargados datos")

        if target_column is None:
            # Usar la última columna como target
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
            logger.info(f"Target: última columna '{self.data.columns[-1]}'")
        else:
            if target_column not in self.data.columns:
                raise ValueError(f"Columna '{target_column}' no existe en el dataset")
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            logger.info(f"Target: columna '{target_column}'")

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def describe(self) -> pd.DataFrame:
        """
        Retorna estadísticas descriptivas del dataset.

        Returns:
            pd.DataFrame: Estadísticas descriptivas
        """
        if self.data is None:
            raise ValueError("No se han cargado datos")
        return self.data.describe(include='all')

    def reset_data(self):
        """
        Resetea los datos a su estado original.
        """
        if self._original_data is not None:
            self.data = self._original_data.copy()
            logger.info("Datos reseteados al estado original")
        else:
            logger.warning("No hay datos originales para resetear")

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        if self.data is not None:
            return f"DataLoader(file='{self.file_path}', shape={self.data.shape})"
        return f"DataLoader(file='{self.file_path}', no_data_loaded)"

    def __len__(self) -> int:
        """Retorna el número de filas del dataset."""
        if self.data is None:
            return 0
        return len(self.data)
