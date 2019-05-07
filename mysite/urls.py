"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from cmdb import views
from cmdb import link_views
urlpatterns = [
    # url(r'^admin/', admin.site.urls),
    url(r'^home/',views.home),
    url(r'^index/textTag.html',views.textTag),
    url(r'^index/textSeg.html',views.textSeg),
    url(r'^index/textStruct.html',views.textStruct),
    url(r'^index/img2Text.html',views.img2Text),
    url(r'^index/imgClusterSeg.html', views.imgClusterSeg),
    url(r'^index/others.html',views.others),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

