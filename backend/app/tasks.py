import datetime
import os.path
from datetime import datetime

import celery.states
from django.core.files.uploadedfile import UploadedFile
import tempfile
import random
import time
from backend.celery import app
from celery import shared_task
from celery.schedules import crontab

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from . import consumers
import os.path as path
from pathlib import Path
import shutil
from django.conf import  settings
from tqdm import tqdm
from django_celery_results.models import TaskResult
from .aniref.clusterer import RefClusterer
from . import models, messages
# from .messages import send_progress_task_msg
from urllib.parse import urljoin
import threading

# @shared_task(name="long_task")
# def long_task(task_id,sec=10):
#     print(f"Task {task_id} started. Sleeping for {sec} seconds")
#     time.sleep(sec)
#     print(f"Task completed!")
#     return f"Task {task_id} finished after {sec} seconds"
# @shared_task(name="beat_task")
# def beat_task():
#     print(f"Running task on {datetime.now()}")
#     time.sleep(3)
#     print(f"Task completed!")
#
# @app.task(bind=True)
# def cleanup_tasks_aniref(self):
#     failed_results = TaskResult.objects.filter(status=celery.states.FAILURE)
#     failed_ids = failed_results.values_list('task_id',flat=True).distinct()
#     failed_tasks = models.AnirefTask.objects.filter(task_id__in=failed_ids)
#     if len(failed_tasks)==0:
#         print("No failed tasks.")
#         return
#     else:
#         print("Failed tasks:",failed_tasks)
#         msg = failed_tasks.delete()
#         print(msg)
#
#
# @app.task(bind=True)
# def progress_task(self):
#     task_id = self.request.id
#     print(f"Running progress task {task_id}")
#     progress = 0
#     state = "STARTED"
#     messages.send_progress_task_msg(progress,state,task_id)
#
#     state = "IN PROGRESS"
#     while progress < 100:
#         time.sleep(5)
#         progress += 10
#         print("Sending progress message")
#         self.update_state(state=state,meta={"progress":progress})
#         messages.send_progress_task_msg(progress,state,task_id)
#     state = "SUCCESS"
#     self.update_state(state=state, meta={"progress": progress})
#     messages.send_progress_task_msg(progress, state, task_id)
#
#     return "Task completed"



aniref_task_data = threading.local()

@app.task(bind=True)
def aniref_task(self,task_info):
    try:
        img_path, vid_path = task_info['img_path'], task_info['vid_path']
        pk = task_info['task_result_pk']
        task_id = self.request.id
        aniref_task_data.task_id = task_id # globally accessible
        print(task_id)
        print(aniref_task_data,aniref_task_data.task_id)
        task_dir = Path(f"/app/media/tasks/{task_id}")
        print(f"Created task directory at {task_dir}")
        if not task_dir.exists():
            task_dir.mkdir(parents=True)
        instance = models.AnirefTask.objects.get(pk=pk)
        instance.ref_img=move_file_field(instance.ref_img,task_dir)
        instance.video=move_file_field(instance.video, task_dir)
        img_path = instance.ref_img.path
        vid_path = instance.video.path
        instance.save()
        print(instance)
        print(instance.ref_img.url)
        print(instance.video.url)
        print(instance.ref_img.path)
        print(instance.video.path)

        temp_dir = task_dir.joinpath("temp")

        messages.update_aniref_task(task_id,info={
            "message":"Task Started!"
        })
        clusterer = RefClusterer(temp_dir=temp_dir)

        messages.update_aniref_task(task_id, info={
            "message": "Start extracting!"
        })
        target_img, sim_imgs = clusterer.extract_similar_from_vid(img_path,vid_path) # do the crucial work

        # print(target_img,sim_imgs)
        target_dir = task_dir.joinpath("results")
        messages.update_aniref_task(task_id, info={
            "message": "Saving results..."
        })
        clusterer.save_similar(target_dir,target_img,sim_imgs)
        sim_dir = target_dir.joinpath("similar_images")
        # print(sim_dir.resolve())

        for img in tqdm(sim_dir.iterdir()):
            if img.is_file():
                img_path = str(img.resolve().relative_to(Path(settings.MEDIA_ROOT)))
                proc_img = models.ProcessedImage(task=instance,img=img_path)
                proc_img.save()
        messages.update_aniref_task(task_id, info={
            "message": "Task completed!",
            "status": celery.states.SUCCESS
        })
    except Exception as e:
        print(e)
        messages.update_aniref_task(task_id, info={
            "message": f"Task failed: {str(e)}",
            "status": celery.states.FAILURE
        })


def move_file_field(field,target_dir):
    media_root = Path(settings.MEDIA_ROOT)
    new_path = Path(target_dir).joinpath(Path(field.path).name)
    field.storage.save(new_path, open(field.path, "rb"))
    field = str(Path(new_path).relative_to(media_root))
    return field
def get_static_url(path:Path,base_url):
    media_root = settings.MEDIA_ROOT
    media_url = settings.MEDIA_URL
    # if isinstance(path,Path):
    media_root = Path(media_root)
    rel_path = path.relative_to(media_root)
    url = urljoin(base_url,media_url.lstrip("/"))+str(rel_path)
    return url


