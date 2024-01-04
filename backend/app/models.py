from django.db import models
from django.contrib.postgres.fields import ArrayField
from django_celery_results.models import TaskResult
from django.contrib.auth import get_user_model
# Create your models here.

User = get_user_model()

class AnirefTask(models.Model):
    name = models.CharField(max_length=255,null=True,blank=True)
    user = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    task_id = models.CharField(max_length=255,null=True,blank=True,unique=True)
    ref_img = models.ImageField(upload_to="temp",null=False, blank=False)
    video = models.FileField(upload_to="temp",null=False, blank=False)
    def __str__(self):
        return (f"Task ID: {self.task_id} \n"
                f"User: {self.user} \n"
                f"Ref Img: {self.ref_img} \n"
                f"Video: {self.video} \n"
                f"Results: {self.results} \n")
class ProcessedImage(models.Model):
    task = models.ForeignKey(AnirefTask,on_delete=models.CASCADE,null=False,blank=False,related_name="results")
    img = models.ImageField(upload_to="results",null=False, blank=False)

'''
What do we need? 
1. Ref Image URL 
2. Video URL (maybe not?) 
3. Result images URL 
4. User(optional) 

'''
