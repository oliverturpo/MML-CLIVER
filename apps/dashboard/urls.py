from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('dashboard/<int:perfil_id>/', views.dashboard_estudiante, name='dashboard_estudiante'),
    path('dashboard/<int:perfil_id>/inicio/', views.dashboard_inicio, name='dashboard_inicio'),
    path('dashboard/<int:perfil_id>/datos/', views.dashboard_datos, name='dashboard_datos'),
    path('dashboard/<int:perfil_id>/modelo/', views.dashboard_modelo, name='dashboard_modelo'),
    path('dashboard/<int:perfil_id>/resultados/', views.dashboard_resultados, name='dashboard_resultados'),
    path('dashboard/<int:perfil_id>/config/', views.dashboard_config, name='dashboard_config'),
    path('seleccionar-algoritmo/<int:perfil_id>/', views.seleccionar_algoritmo, name='seleccionar_algoritmo'),
]