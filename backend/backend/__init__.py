from .celery import app as celery_app
__all__ = ('celery_app') # make sure the app is loaded when Django starts