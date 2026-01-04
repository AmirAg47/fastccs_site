import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ccspredictor.settings')

app = Celery('ccspredictor')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
app.conf.task_send_sent_event = True
task_time_limit = 300  # تایم‌اوت کل تسک برحسب ثانیه
task_soft_time_limit = 250  # اخطار نرم‌افزاری قبل از تایم‌اوت سخت
