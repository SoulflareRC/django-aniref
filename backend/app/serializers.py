
from rest_framework import serializers
from django_celery_results.models import TaskResult
from . import models

class ProcessedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ProcessedImage
        fields = "__all__"

class TaskResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaskResult
        fields = "__all__"

class AnirefTaskSerializer(serializers.ModelSerializer):
    results = ProcessedImageSerializer(many=True,read_only=True)
    task_result = serializers.SerializerMethodField()
    class Meta:
        model = models.AnirefTask
        fields = "__all__"
    def create(self, validated_data):
        print(validated_data)
        return super().create(validated_data)
    def get_task_result(self,instance):
        task_result = models.TaskResult.objects.get_task(instance.task_id)
        return TaskResultSerializer(task_result,context=self.context).data

