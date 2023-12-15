from django.contrib import admin
from django.urls import path, include
from . import views, tasks
urlpatterns = [
    path("test/",views.TestView.as_view()),
    path("progress/",views.ProgressView.as_view()),
    path("chat/", views.index, name="index"),
    path("chat/<str:room_name>/", views.room, name="room"),
    path("api/progress/",views.StartProgressTaskView.as_view(),name="api_progress"),
    path("api/task/state/",views.TaskStatusView.as_view(),name="task_state"),
    path("api/test/",views.TestAPIView.as_view(),name="api_test"),
]
