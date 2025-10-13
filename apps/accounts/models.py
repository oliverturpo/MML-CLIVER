from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone

class PerfilEstudiante(models.Model):
    """
    Modelo que representa el perfil de cada estudiante.
    Cada estudiante tiene su propio espacio protegido con PIN de 4 dígitos.
    """
    
    # Información personal
    nombre_completo = models.CharField(
        max_length=150,
        verbose_name="Nombre Completo"
    )
    
    email = models.EmailField(
        unique=True,
        verbose_name="Correo Electrónico"
    )
    
    carrera = models.CharField(
        max_length=200,
        verbose_name="Carrera",
        default="Ingeniería Estadística e Informática"
    )
    
    # PIN de 4 dígitos para acceso
    pin_validator = RegexValidator(
        regex=r'^\d{4}$',
        message='El PIN debe contener exactamente 4 dígitos numéricos'
    )
    
    pin = models.CharField(
        max_length=4,
        validators=[pin_validator],
        verbose_name="PIN de Acceso"
    )
    
    # Avatar/Foto (opcional)
    avatar = models.ImageField(
        upload_to='avatars/',
        null=True,
        blank=True,
        verbose_name="Foto de Perfil"
    )
    
    # Algoritmo elegido
    ALGORITMOS_CHOICES = [
        ('regresion_lineal', 'Regresión Lineal'),
        ('regresion_logistica', 'Regresión Logística'),
        ('ridge_lasso', 'Ridge y Lasso'),
        ('arbol_cart', 'Árbol de Regresión CART'),
        ('knn', 'K-Nearest Neighbors Regression'),
        ('red_neuronal', 'Red Neuronal para Regresión'),
    ]
    
    algoritmo_elegido = models.CharField(
        max_length=50,
        choices=ALGORITMOS_CHOICES,
        verbose_name="Algoritmo Elegido",
        null=True,
        blank=True
    )
    
    # Dataset
    dataset_nombre = models.CharField(
        max_length=200,
        verbose_name="Nombre del Dataset",
        null=True,
        blank=True
    )
    
    dataset_archivo = models.FileField(
        upload_to='datasets/',
        null=True,
        blank=True,
        verbose_name="Archivo Dataset"
    )
    
    # Métricas del modelo (guardadas como JSON)
    metricas = models.JSONField(
        null=True,
        blank=True,
        verbose_name="Métricas del Modelo"
    )
    
    # Modelo entrenado guardado
    modelo_archivo = models.FileField(
        upload_to='modelos/',
        null=True,
        blank=True,
        verbose_name="Modelo Entrenado"
    )
    
    # Metadatos
    fecha_registro = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Registro"
    )
    
    fecha_actualizacion = models.DateTimeField(
        auto_now=True,
        verbose_name="Última Actualización"
    )
    
    activo = models.BooleanField(
        default=True,
        verbose_name="Perfil Activo"
    )
    
    class Meta:
        verbose_name = "Perfil de Estudiante"
        verbose_name_plural = "Perfiles de Estudiantes"
        ordering = ['-fecha_registro']
    
    def __str__(self):
        return f"{self.nombre_completo} - {self.get_algoritmo_elegido_display() or 'Sin algoritmo'}"
    
    def tiene_modelo_entrenado(self):
        """Verifica si el estudiante ya entrenó su modelo"""
        return bool(self.modelo_archivo and self.metricas)
    
    def obtener_metrica(self, nombre_metrica):
        """Obtiene una métrica específica del modelo"""
        if self.metricas and nombre_metrica in self.metricas:
            return self.metricas[nombre_metrica]
        return None


class CodigoAcceso(models.Model):
    """
    Modelo para gestionar códigos de acceso para registro.
    Solo estudiantes con código válido pueden registrarse.
    """
    
    codigo = models.CharField(
        max_length=50,
        unique=True,
        verbose_name="Código de Acceso"
    )
    
    descripcion = models.CharField(
        max_length=200,
        verbose_name="Descripción",
        blank=True
    )
    
    usos_maximos = models.IntegerField(
        default=1,
        verbose_name="Usos Máximos",
        help_text="Número máximo de veces que puede usarse este código. 0 = ilimitado"
    )
    
    usos_actuales = models.IntegerField(
        default=0,
        verbose_name="Usos Actuales"
    )
    
    fecha_expiracion = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Fecha de Expiración"
    )
    
    activo = models.BooleanField(
        default=True,
        verbose_name="Código Activo"
    )
    
    fecha_creacion = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Fecha de Creación"
    )
    
    class Meta:
        verbose_name = "Código de Acceso"
        verbose_name_plural = "Códigos de Acceso"
        ordering = ['-fecha_creacion']
    
    def __str__(self):
        return f"{self.codigo} ({self.usos_actuales}/{self.usos_maximos if self.usos_maximos > 0 else '∞'})"
    
    def es_valido(self):
        """Verifica si el código es válido para usar"""
        if not self.activo:
            return False
        
        if self.fecha_expiracion and timezone.now() > self.fecha_expiracion:
            return False
        
        if self.usos_maximos > 0 and self.usos_actuales >= self.usos_maximos:
            return False
        
        return True
    
    def usar_codigo(self):
        """Incrementa el contador de usos del código"""
        self.usos_actuales += 1
        self.save()