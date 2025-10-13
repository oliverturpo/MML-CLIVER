from django.contrib import admin
from .models import PerfilEstudiante, CodigoAcceso

@admin.register(PerfilEstudiante)
class PerfilEstudianteAdmin(admin.ModelAdmin):
    list_display = ['nombre_completo', 'email', 'algoritmo_elegido', 'fecha_registro', 'activo']
    list_filter = ['algoritmo_elegido', 'activo', 'fecha_registro']
    search_fields = ['nombre_completo', 'email']
    readonly_fields = ['fecha_registro', 'fecha_actualizacion']

@admin.register(CodigoAcceso)
class CodigoAccesoAdmin(admin.ModelAdmin):
    list_display = ['codigo', 'usos_actuales', 'usos_maximos', 'fecha_expiracion', 'activo']
    list_filter = ['activo', 'fecha_creacion']
    search_fields = ['codigo', 'descripcion']
    readonly_fields = ['fecha_creacion', 'usos_actuales']