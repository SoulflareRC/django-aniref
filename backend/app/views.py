import time

from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
# Create your views here.
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from . import tasks
class TestView(View):
    def get(self,request,*args,**kwargs):
        q = request.GET.get('q')
        # print(f"Working on task {q} for {20} seconds(no celery)")
        # # time.sleep(20)
        # print(f"Task {q} completed")
        # now = datetime.now()
        task = tasks.long_task.delay(q,20)
        return HttpResponse(f"We are processing task {task.id}")
        # return HttpResponse(f"Hello!Now is {now}")
class ProgressView(View):
    def get(self,request,*args,**kwargs):
        q = request.GET.get('q')
        # task = tasks.progress_task.delay("progress")
        # print(f"We are working on the progress task {task.id}")
        return render(request,"progress.html")
    def post(self,request):
        tasks.progress_task.delay()
        return HttpResponse("Task launched!")
class StartProgressTaskView(APIView):
    def post(self,request):
        progress_task = tasks.progress_task.delay()
        return Response(
            data={
                "message":f"Progress task {progress_task.id} created!",
                "task_id":progress_task.id
            }
        )

class TaskStatusView(APIView):
    def get(self, request):
        task_id = request.GET.get('task_id')
        task_type = request.GET.get('task_type')
        if not task_id or not task_type:
            return Response({'error': 'Task ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            task_func = getattr(tasks, task_type)
            result = task_func.AsyncResult(task_id)
            task_state = result.state
            task_info = result.info
            return Response({'task_id': task_id, 'state': task_state, "info":task_info}, status=status.HTTP_200_OK)
        except Exception as e:
            print("Error:",e)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TestAPIView(APIView):
    def get(self,request):
        print(request.user)
        return  Response({"message":"test"})
'''
so a possible workflow: 
frontend: 
/task/:trigger submit a task with {task_id}, immediately redirect to /task/{task_id}  
/task/{task_id}:
1.1:After redirection Open websocket connection, receive task progress by websocket 
1.2:No redirect, directly go to this page after task started,


'''
# chat/views.py
from django.shortcuts import render
def index(request):
    return render(request, "index.html")

def room(request, room_name):
    return render(request, "room.html", {"room_name": room_name})
