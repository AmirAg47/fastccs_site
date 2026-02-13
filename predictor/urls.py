from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect 
from . import views
from django.http import HttpResponse


urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('upload/', views.upload, name='upload'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
 
 
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
from django.http import HttpResponse

def certum_verify(request):
    return HttpResponse(
        "99e5f8d60e442a2e2071634522399e2886d828516ec5afe17d29d11493a44b8-certum.pl",
        content_type="text/plain"
    )

urlpatterns += [
    path('.well-known/pki-validation/certum.txt', certum_verify),
]
