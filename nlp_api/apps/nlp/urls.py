from django.urls import path

from . import views

urlpatterns = [
    path('<str:model_name>/',views.index)
]
