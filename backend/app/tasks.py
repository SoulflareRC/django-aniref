import random
import time
from backend.celery import app
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from . import consumers

@shared_task
def add(x, y):
    # Celery recognizes this as the `movies.tasks.add` task
    # the name is purposefully omitted here.
    return x + y

@shared_task(name="multiply_two_numbers")
def mul(x, y):
    # Celery recognizes this as the `multiple_two_numbers` task
    total = x * (y * random.randint(3, 100))
    return total

@shared_task(name="sum_list_numbers")
def xsum(numbers):
    # Celery recognizes this as the `sum_list_numbers` task
    return sum(numbers)
@shared_task(name="long_task")
def long_task(task_id,sec=10):
    print(f"Task {task_id} started. Sleeping for {sec} seconds")
    time.sleep(sec)
    print(f"Task completed!")
    return f"Task {task_id} finished after {sec} seconds"
@shared_task(name="beat_task")
def beat_task(task_id):
    print(f"Running task {task_id}")
    time.sleep(3)
    print(f"Task {task_id} completed!")

# @shared_task(name="progress_task")

@app.task(bind=True)
def progress_task(self):
    task_id = self.request.id
    print(f"Running progress task {task_id}")
    progress = 0
    state = "STARTED"
    send_progress_task_msg(progress,state,task_id)

    state = "IN PROGRESS"
    while progress < 100:
        time.sleep(5)
        progress += 10
        print("Sending progress message")
        self.update_state(state=state,meta={"progress":progress})
        send_progress_task_msg(progress,state,task_id)
    state = "SUCCESS"
    self.update_state(state=state, meta={"progress": progress})
    send_progress_task_msg(progress, state, task_id)

    return "Task completed"

def send_progress_task_msg(progress,state,task_id):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "progress_task_group", {
            "type": "progress.message",
            "data": {
                "task_id": task_id,
                "state": state,
                "progress": progress,
            }
        }
    )
