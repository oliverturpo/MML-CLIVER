from django.urls import path
from . import views

urlpatterns = [
    path('registro/', views.registro_view, name='registro'),
    path('login/<int:perfil_id>/', views.login_pin_view, name='login_pin'),
    path('logout/', views.logout_view, name='logout'),
]