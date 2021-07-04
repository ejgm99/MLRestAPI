from django.urls import path

from . import views

urlpatterns = [
    path('<str:twitterQuery>/',views.evaluateTopic),
    path('',views.evaluateTopics)
]
