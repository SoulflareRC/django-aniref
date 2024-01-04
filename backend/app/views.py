import time
from django.core.files.uploadedfile import UploadedFile
from django.core.files import File
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
# Create your views here.
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from pathlib import  Path
from django.conf import settings
from django.core.files import File
from django_celery_results.models import TaskResult
import os.path
from . import tasks,models,serializers
# class TestView(View):
#     def get(self,request,*args,**kwargs):
#         q = request.GET.get('q')
#         # print(f"Working on task {q} for {20} seconds(no celery)")
#         # # time.sleep(20)
#         # print(f"Task {q} completed")
#         # now = datetime.now()
#         task = tasks.long_task.delay(q,20)
#         return HttpResponse(f"We are processing task {task.id}")
#         # return HttpResponse(f"Hello!Now is {now}")
# class ProgressView(View):
#     def get(self,request,*args,**kwargs):
#         q = request.GET.get('q')
#         # task = tasks.progress_task.delay("progress")
#         # print(f"We are working on the progress task {task.id}")
#         return render(request,"progress.html")
#     def post(self,request):
#         tasks.progress_task.delay()
#         return HttpResponse("Task launched!")
# class StartProgressTaskView(APIView):
#     def post(self,request):
#         progress_task = tasks.progress_task.delay()
#         return Response(
#             data={
#                 "message":f"Progress task {progress_task.id} created!",
#                 "task_id":progress_task.id
#             }
#         )
#
# class TaskStatusView(APIView):
#     def get(self, request):
#         task_id = request.GET.get('task_id')
#         task_type = request.GET.get('task_type')
#         if not task_id or not task_type:
#             return Response({'error': 'Task ID is required'}, status=status.HTTP_400_BAD_REQUEST)
#         try:
#             task_func = getattr(tasks, task_type)
#             result = task_func.AsyncResult(task_id)
#             task_state = result.state
#             task_info = result.info
#             return Response({'task_id': task_id, 'state': task_state, "info":task_info}, status=status.HTTP_200_OK)
#         except Exception as e:
#             print("Error:",e)
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
# class AniRefTaskView(APIView):
#     def post(self, request):
#         try:
#             img_file: UploadedFile = request.FILES['ref_img']
#             vid_file: UploadedFile = request.FILES['vid']
#
#             # Save uploaded files to a permanent location
#             temp_dir = Path(settings.MEDIA_ROOT).joinpath("temp")
#             if not temp_dir.exists():
#                 temp_dir.mkdir(parents=True)
#             img_path = temp_dir.joinpath(img_file.name)
#             vid_path = temp_dir.joinpath(vid_file.name)
#
#
#             with open(img_path, 'wb') as img_dest, open(vid_path, 'wb') as vid_dest:
#                 for chunk in img_file.chunks():
#                     img_dest.write(chunk)
#                 for chunk in vid_file.chunks():
#                     vid_dest.write(chunk)
#
#             # Pass file paths to Celery tasks
#             task = tasks.aniref_task.delay(str(img_path.resolve()),str(vid_path.resolve()))
#
#             return Response(data={
#                 "message": f"Aniref task {task.id} created!",
#                 "task_id": task.id
#             })
#
#         except KeyError:
#             return Response({"error": "Please provide both 'ref_img' and 'vid' files"}, status=400)
#         except Exception as e:
#             return Response({"error": str(e)}, status=500)
class AnirefTaskViewset(ModelViewSet):
    queryset = models.AnirefTask.objects.all()
    serializer_class = serializers.AnirefTaskSerializer
    def get_queryset(self):
        qs = super().get_queryset()
        task_id = self.request.query_params.get("task_id")
        user = self.request.query_params.get("user")
        if user:
            qs = qs.filter(user=user)
        if task_id:
            qs = qs.filter(task_id=task_id)
        return qs
    def create(self, request, *args, **kwargs):
        response = super().create( request, *args, **kwargs)

        instance = models.AnirefTask.objects.get(pk=response.data['id'])

        media_root = Path(settings.MEDIA_ROOT)

        # new_path = "/app/media/test/" + Path(instance.ref_img.path).name
        # instance.ref_img.storage.save(new_path,open(instance.ref_img.path, "rb"))
        # instance.ref_img = str(Path(new_path).relative_to(media_root))
        # instance.save()
        # print(instance)
        # print(instance.ref_img.path)
        # print(instance.video.path)
        # print(instance.ref_img.storage)

        task_info = {
            "img_path": str(instance.ref_img.path),
            "vid_path": str(instance.video.path),
            "task_result_pk": instance.pk
        }
        aniref_task = tasks.aniref_task.delay(task_info)
        # task_result = TaskResult.objects.get_task(aniref_task.id)
        # task_result.save()
        # instance.task = task_result
        instance.task_id = str(aniref_task.id)
        instance.save()
        #
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

        # return response

        # try:
        #     img_file: UploadedFile = request.FILES['ref_img']
        #     vid_file: UploadedFile = request.FILES['vid']
        #
        #     # Save uploaded files to a permanent location
        #     temp_dir = Path(settings.MEDIA_ROOT).joinpath("temp")
        #     if not temp_dir.exists():
        #         temp_dir.mkdir(parents=True)
        #     img_path = temp_dir.joinpath(img_file.name)
        #     vid_path = temp_dir.joinpath(vid_file.name)
        #
        #
        #     with open(img_path, 'wb') as img_dest, open(vid_path, 'wb') as vid_dest:
        #         for chunk in img_file.chunks():
        #             img_dest.write(chunk)
        #         for chunk in vid_file.chunks():
        #             vid_dest.write(chunk)
        #
        #     # Pass file paths to Celery tasks
        #     base_url = request.build_absolute_uri("/")
        #     task_info = {
        #         "img_path": str(img_path.resolve()),
        #         "vid_path": str(vid_path.resolve()),
        #         "base_url": base_url,
        #     }
        #     task = tasks.aniref_task.delay(task_info)
        #
        #     return Response(data={
        #         "message": f"Aniref task {task.id} created!",
        #         "task_id": task.id
        #     })
        #
        # except KeyError:
        #     return Response({"error": "Please provide both 'ref_img' and 'vid' files"}, status=400)
        # except Exception as e:
        #     return Response({"error": str(e)}, status=500)


# class TestAPIView(APIView):
#     def get(self,request):
#         print(request.user)
#         return  Response({"message":"test"})
# '''
# so a possible workflow:
# frontend:
# /task/:trigger submit a task with {task_id}, immediately redirect to /task/{task_id}
# /task/{task_id}:
# 1.1:After redirection Open websocket connection, receive task progress by websocket
# 1.2:No redirect, directly go to this page after task started,
#
#
# '''
# # chat/views.py
# from django.shortcuts import render
# def index(request):
#     return render(request, "index.html")
#
# def room(request, room_name):
#     return render(request, "room.html", {"room_name": room_name})
