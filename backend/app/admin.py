from django.contrib import admin
from django_celery_results.models import TaskResult
from . import models
# Register your models here.
class AnirefTaskAdmin(admin.ModelAdmin):
    list_display = ["task_id","user"]
    fields = ["task_id","user","ref_img","video","results"]
class ProcessedImageAdmin(admin.ModelAdmin):
    list_display = ['task','img']
    fields = ['task','img']

admin.site.register(models.AnirefTask, AnirefTaskAdmin)