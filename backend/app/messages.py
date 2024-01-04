from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import AnirefTask

# def send_progress_task_msg(progress,state,task_id):
#     channel_layer = get_channel_layer()
#     async_to_sync(channel_layer.group_send)(
#         "progress_task_group", {
#             "type": "progress.message",
#             "data": {
#                 "task_id": task_id,
#                 "state": state,
#                 "progress": progress,
#             }
#         }
#     )
def update_aniref_task(task_id,info):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"task_aniref_{task_id}",{
            "type":"task_progress",
            "info":info,
        }
    )