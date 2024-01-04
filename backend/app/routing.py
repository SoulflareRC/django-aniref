# chat/routing.py
from django.urls import re_path, path

from . import consumers

websocket_urlpatterns = [
    # re_path(r"ws/app/chat/(?P<room_name>\w+)/$", consumers.ChatConsumer.as_asgi()),
    # path(r"ws/app/progress/",consumers.TaskConsumer.as_asgi()),
    path(r"ws/app/tasks/aniref/<str:task_id>/",consumers.AnirefTaskUpdateConsumer.as_asgi()),
]