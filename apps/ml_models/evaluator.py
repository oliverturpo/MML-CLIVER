"""
Módulo ModelEvaluator: Evaluación y métricas de modelos ML
Implementa principios POO: Encapsulamiento, Abstracción y Single Responsibility Principle
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Clase responsable de evaluar modelos y calcular métricas.

    Principios POO aplicados:
    - Encapsulamiento: Mantiene el estado de las evaluaciones
    - Abstracción: Oculta la complejidad del cálculo de métricas
    - Single Responsibility: Solo se encarga de evaluar modelos

    Attributes:
        problem_type: Tipo de problema ('classification' o 'regression')
        metrics: Diccionario con las métricas calculadas
        y_true: Valores reales
        y_pred: Valores predichos
    """

    def __init__(self, problem_type: str = 'regression'):
        """
        Inicializa el ModelEvaluator.

        Args:
            problem_type (str): 'regression' o 'classification'
        """
        self.problem_type = problem_type
        self.metrics: Dict = {}
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_pred_proba: Optional[np.ndarray] = None

        logger.info(f"ModelEvaluator inicializado para {problem_type}")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Evalúa el modelo y calcula todas las métricas relevantes.

        Args:
            y_true (np.ndarray): Valores reales
            y_pred (np.ndarray): Valores predichos
            y_pred_proba (np.ndarray, optional): Probabilidades predichas (para clasificación)

        Returns:
            Dict: Diccionario con todas las métricas
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        logger.info(f"Evaluando modelo de {self.problem_type}...")

        if self.problem_type == 'regression':
            self.metrics = self._calculate_regression_metrics()
        else:
            self.metrics = self._calculate_classification_metrics()

        logger.info("Evaluación completada")
        return self.metrics

    def _calculate_regression_metrics(self) -> Dict:
        """Calcula métricas para problemas de regresión."""
        metrics = {}

        # MSE (Mean Squared Error)
        mse = mean_squared_error(self.y_true, self.y_pred)
        metrics['MSE'] = float(mse)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        metrics['RMSE'] = float(rmse)

        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        metrics['MAE'] = float(mae)

        # R² Score
        r2 = r2_score(self.y_true, self.y_pred)
        metrics['R2_Score'] = float(r2)

        # MAPE (Mean Absolute Percentage Error)
        # Evitar división por cero
        mask = self.y_true != 0
        if mask.any():
            mape = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
            metrics['MAPE'] = float(mape)

        # Adjusted R² (si se proporciona el número de features)
        # metrics['Adjusted_R2'] = self._calculate_adjusted_r2(r2, n_samples, n_features)

        logger.info(f"Métricas de regresión: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        return metrics

    def _calculate_classification_metrics(self) -> Dict:
        """Calcula métricas para problemas de clasificación."""
        metrics = {}

        # Accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        metrics['Accuracy'] = float(accuracy)

        # Precision, Recall, F1-Score
        # Determinar si es binario o multiclase
        n_classes = len(np.unique(self.y_true))

        if n_classes == 2:
            # Clasificación binaria
            precision = precision_score(self.y_true, self.y_pred, zero_division=0)
            recall = recall_score(self.y_true, self.y_pred, zero_division=0)
            f1 = f1_score(self.y_true, self.y_pred, zero_division=0)

            metrics['Precision'] = float(precision)
            metrics['Recall'] = float(recall)
            metrics['F1_Score'] = float(f1)

            # ROC-AUC si hay probabilidades
            if self.y_pred_proba is not None:
                # Si y_pred_proba es 2D, tomar la segunda columna (clase positiva)
                if len(self.y_pred_proba.shape) > 1 and self.y_pred_proba.shape[1] > 1:
                    proba = self.y_pred_proba[:, 1]
                else:
                    proba = self.y_pred_proba

                try:
                    roc_auc = roc_auc_score(self.y_true, proba)
                    metrics['ROC_AUC'] = float(roc_auc)

                    # Calcular curva ROC
                    fpr, tpr, thresholds = roc_curve(self.y_true, proba)
                    metrics['ROC_FPR'] = fpr.tolist()
                    metrics['ROC_TPR'] = tpr.tolist()
                    metrics['ROC_Thresholds'] = thresholds.tolist()
                except Exception as e:
                    logger.warning(f"No se pudo calcular ROC-AUC: {str(e)}")

        else:
            # Clasificación multiclase - promedios
            precision = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)

            metrics['Precision_Weighted'] = float(precision)
            metrics['Recall_Weighted'] = float(recall)
            metrics['F1_Score_Weighted'] = float(f1)

        # Matriz de Confusión
        cm = confusion_matrix(self.y_true, self.y_pred)
        metrics['Confusion_Matrix'] = cm.tolist()

        # Reporte de clasificación
        report = classification_report(self.y_true, self.y_pred, output_dict=True, zero_division=0)
        metrics['Classification_Report'] = report

        logger.info(f"Métricas de clasificación: Accuracy={accuracy:.4f}")

        return metrics

    def calculate_adjusted_r2(self, n_samples: int, n_features: int) -> float:
        """
        Calcula el R² ajustado.

        Args:
            n_samples (int): Número de muestras
            n_features (int): Número de características

        Returns:
            float: R² ajustado
        """
        if 'R2_Score' not in self.metrics:
            raise ValueError("Debe calcular R² primero")

        r2 = self.metrics['R2_Score']
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        self.metrics['Adjusted_R2'] = float(adjusted_r2)

        return adjusted_r2

    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                             class_names: Optional[list] = None) -> str:
        """
        Genera un gráfico de la matriz de confusión.

        Args:
            save_path (str, optional): Ruta donde guardar la imagen
            class_names (list, optional): Nombres de las clases

        Returns:
            str: Imagen en formato base64 si no se especifica save_path
        """
        if 'Confusion_Matrix' not in self.metrics:
            raise ValueError("Debe calcular métricas de clasificación primero")

        cm = np.array(self.metrics['Confusion_Matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or 'auto',
                   yticklabels=class_names or 'auto')
        plt.title('Matriz de Confusión')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada en: {save_path}")
            plt.close()
            return save_path
        else:
            # Retornar como base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64

    def plot_roc_curve(self, save_path: Optional[str] = None) -> str:
        """
        Genera un gráfico de la curva ROC.

        Args:
            save_path (str, optional): Ruta donde guardar la imagen

        Returns:
            str: Imagen en formato base64 si no se especifica save_path
        """
        if 'ROC_FPR' not in self.metrics or 'ROC_TPR' not in self.metrics:
            raise ValueError("Debe calcular métricas ROC primero")

        fpr = self.metrics['ROC_FPR']
        tpr = self.metrics['ROC_TPR']
        roc_auc = self.metrics.get('ROC_AUC', auc(fpr, tpr))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea Base')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC (Receiver Operating Characteristic)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Curva ROC guardada en: {save_path}")
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64

    def plot_predictions(self, save_path: Optional[str] = None) -> str:
        """
        Genera un gráfico de predicciones vs valores reales (para regresión).

        Args:
            save_path (str, optional): Ruta donde guardar la imagen

        Returns:
            str: Imagen en formato base64 si no se especifica save_path
        """
        if self.problem_type != 'regression':
            raise ValueError("Este método es solo para problemas de regresión")

        plt.figure(figsize=(10, 6))

        # Scatter plot
        plt.scatter(self.y_true, self.y_pred, alpha=0.5, edgecolors='k', s=50)

        # Línea de predicción perfecta
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')

        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title('Predicciones vs Valores Reales')
        plt.legend()
        plt.grid(alpha=0.3)

        # Añadir métricas al gráfico
        r2 = self.metrics.get('R2_Score', 0)
        rmse = self.metrics.get('RMSE', 0)
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico de predicciones guardado en: {save_path}")
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64

    def plot_residuals(self, save_path: Optional[str] = None) -> str:
        """
        Genera un gráfico de residuos (para regresión).

        Args:
            save_path (str, optional): Ruta donde guardar la imagen

        Returns:
            str: Imagen en formato base64 si no se especifica save_path
        """
        if self.problem_type != 'regression':
            raise ValueError("Este método es solo para problemas de regresión")

        residuals = self.y_true - self.y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Residuos vs Predicciones
        ax1.scatter(self.y_pred, residuals, alpha=0.5, edgecolors='k', s=50)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Valores Predichos')
        ax1.set_ylabel('Residuos')
        ax1.set_title('Residuos vs Predicciones')
        ax1.grid(alpha=0.3)

        # Histograma de residuos
        ax2.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        ax2.set_xlabel('Residuos')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Residuos')
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Gráfico de residuos guardado en: {save_path}")
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64

    def get_metrics(self) -> Dict:
        """Retorna todas las métricas calculadas."""
        return self.metrics

    def get_summary(self) -> str:
        """
        Retorna un resumen en texto de las métricas.

        Returns:
            str: Resumen de métricas
        """
        if not self.metrics:
            return "No hay métricas calculadas"

        summary = f"\n{'='*50}\n"
        summary += f"RESUMEN DE EVALUACIÓN - {self.problem_type.upper()}\n"
        summary += f"{'='*50}\n\n"

        if self.problem_type == 'regression':
            summary += f"MSE (Error Cuadrático Medio): {self.metrics.get('MSE', 'N/A'):.4f}\n"
            summary += f"RMSE (Raíz del MSE): {self.metrics.get('RMSE', 'N/A'):.4f}\n"
            summary += f"MAE (Error Absoluto Medio): {self.metrics.get('MAE', 'N/A'):.4f}\n"
            summary += f"R² Score: {self.metrics.get('R2_Score', 'N/A'):.4f}\n"
            if 'MAPE' in self.metrics:
                summary += f"MAPE: {self.metrics.get('MAPE', 'N/A'):.2f}%\n"
        else:
            summary += f"Accuracy: {self.metrics.get('Accuracy', 'N/A'):.4f}\n"
            if 'Precision' in self.metrics:
                summary += f"Precision: {self.metrics.get('Precision', 'N/A'):.4f}\n"
                summary += f"Recall: {self.metrics.get('Recall', 'N/A'):.4f}\n"
                summary += f"F1-Score: {self.metrics.get('F1_Score', 'N/A'):.4f}\n"
            if 'ROC_AUC' in self.metrics:
                summary += f"ROC-AUC: {self.metrics.get('ROC_AUC', 'N/A'):.4f}\n"

        summary += f"\n{'='*50}\n"

        return summary

    def __repr__(self) -> str:
        """Representación en string del objeto."""
        n_metrics = len(self.metrics)
        return f"ModelEvaluator(type='{self.problem_type}', metrics={n_metrics})"
