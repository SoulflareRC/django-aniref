import os

from celery import Celery
from decouple import  config
os.environ.setdefault('DJANGO_SETTINGS_MODULE','backend.settings')
app = Celery('aniref')
app.config_from_object('django.conf:settings',namespace="CELERY")
app.autodiscover_tasks()
app.loader.override_backends['django-db'] = 'django_celery_results.backends.database:DatabaseBackend'
