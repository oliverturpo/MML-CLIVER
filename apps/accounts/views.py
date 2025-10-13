from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import PerfilEstudiante, CodigoAcceso

def registro_view(request):
    """Vista para registrar un nuevo estudiante"""
    
    if request.method == 'POST':
        nombre = request.POST.get('nombre_completo')
        email = request.POST.get('email')
        carrera = request.POST.get('carrera')
        pin = request.POST.get('pin')
        codigo_acceso = request.POST.get('codigo_acceso')
        
        # Validar código de acceso
        try:
            codigo = CodigoAcceso.objects.get(codigo=codigo_acceso)
            if not codigo.es_valido():
                messages.error(request, 'El código de acceso no es válido o ha expirado')
                return render(request, 'accounts/registro.html')
        except CodigoAcceso.DoesNotExist:
            messages.error(request, 'Código de acceso incorrecto')
            return render(request, 'accounts/registro.html')
        
        # Validar que el email no exista
        if PerfilEstudiante.objects.filter(email=email).exists():
            messages.error(request, 'Ya existe un perfil con este correo electrónico')
            return render(request, 'accounts/registro.html')
        
        # Validar PIN de 4 dígitos
        if not pin.isdigit() or len(pin) != 4:
            messages.error(request, 'El PIN debe ser exactamente 4 dígitos numéricos')
            return render(request, 'accounts/registro.html')
        
        # Crear perfil
        perfil = PerfilEstudiante.objects.create(
            nombre_completo=nombre,
            email=email,
            carrera=carrera,
            pin=pin
        )
        
        # Usar el código
        codigo.usar_codigo()
        
        messages.success(request, '¡Registro exitoso! Tu perfil ha sido creado')
        return redirect('landing_page')
    
    return render(request, 'accounts/registro.html')


def login_pin_view(request, perfil_id):
    """Vista para acceder a un proyecto con PIN"""
    
    perfil = get_object_or_404(PerfilEstudiante, id=perfil_id)
    
    if request.method == 'POST':
        pin_ingresado = request.POST.get('pin')
        
        if pin_ingresado == perfil.pin:
            # PIN correcto - guardar en sesión
            request.session['perfil_activo'] = perfil_id
            messages.success(request, f'Bienvenido, {perfil.nombre_completo}')
            return redirect('dashboard_estudiante', perfil_id=perfil_id)
        else:
            messages.error(request, 'PIN incorrecto. Intenta de nuevo')
    
    context = {
        'perfil': perfil
    }
    return render(request, 'accounts/login_pin.html', context)


def logout_view(request):
    """Cerrar sesión del perfil activo"""
    
    if 'perfil_activo' in request.session:
        del request.session['perfil_activo']
    
    messages.info(request, 'Sesión cerrada')
    return redirect('landing_page')