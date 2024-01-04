# chat/consumers.py
import json
import celery
from channels.generic.websocket import AsyncWebsocketConsumer
from . import tasks

class AnirefTaskUpdateConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        print(f"Task {self.task_id} connected!")
        self.task_group_name = f"task_aniref_{self.task_id}"
        await self.channel_layer.group_add(
            self.task_group_name,self.channel_name
        )
        await self.accept()
    async def disconnect(self, code):
        await self.channel_layer.group_discard(
            self.task_group_name, self.channel_name
        )
    async def receive(self, text_data=None, bytes_data=None):
        # doesn't need to receive any message
        pass
    async def task_progress(self,event):
        info = event['info']
        await self.send(text_data=json.dumps({
            'info':info
        }))

#
# class TaskConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         # self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
#         self.room_group_name = f"progress_task_group"
#
#         # Join room group
#         await self.channel_layer.group_add(self.room_group_name, self.channel_name)
#
#         await self.accept()
#
#     async def disconnect(self, close_code):
#         # Leave room group
#         await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
#
#     # Receive message from WebSocket
#     async def receive(self, text_data):
#         print("TaskConsumer received message!")
#         print(text_data)
#
#         text_data_json = json.loads(text_data)
#         message = text_data_json["message"]
#         # Send message to room group
#         await self.send(text_data=json.dumps({
#             "message":"Hello!"
#         }))
#         tasks.progress_task("some-id").delay()
#         await self.channel_layer.group_send(
#             self.room_group_name, {"type": "progress.message", "data": {
#                 "message":message
#             }
#         }
#         )
#
#     # Receive message from room group
#     async def progress_message(self, event):
#         message = event["data"]
#         print("Received message:",message)
#
#         # Send message to WebSocket
#         await self.send(text_data=json.dumps({"message": message}))
#
# # Logic to update task status and send WebSocket updates
# async def update_task(task_id, progress):
#     # Send task updates to WebSocket clients
#     result = celery.results.AsyncResult(task_id)
#     await TaskConsumer.send_task_update(task_id, {"progress": progress,"status":result.status})
#
#
# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
#         self.room_group_name = f"chat_{self.room_name}"
#
#         print(f"Joining channel {self.channel_name} to group {self.room_group_name}")
#         # Join room group
#         await self.channel_layer.group_add(self.room_group_name, self.channel_name)
#
#         await self.accept()
#
#     async def disconnect(self, close_code):
#         # Leave room group
#         await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
#
#     # Receive message from WebSocket
#     async def receive(self, text_data):
#         text_data_json = json.loads(text_data)
#         message = text_data_json["message"]
#         print(f"sending message {message} to group {self.room_group_name}")
#         # Send message to room group
#         await self.channel_layer.group_send(
#             self.room_group_name, {"type": "chat.message", "message": message}
#         )
#
#     # Receive message from room group
#     async def chat_message(self, event):
#         message = event["message"]
#
#         # Send message to WebSocket
#         await self.send(text_data=json.dumps({"message": message}))