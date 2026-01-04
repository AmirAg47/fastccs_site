import pandas as pd
import numpy as np
import uuid
import os
import time
import logging
from datetime import timedelta
from predictor.tasks import run_prediction_task
import threading
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from pathlib import Path
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.shortcuts import redirect
from .forms import CSVUploadForm
from .utils import load_encoder, compute_molecular_properties, load_model, predict_data, load_scaler, load_cluster, safe_compute_properties
import tflite_runtime.interpreter as tflite  # For TensorFlow Lite Runtime
from .utils import property_columns
from celery.result import AsyncResult
from django.http import JsonResponse
from django.core.cache import cache

logger = logging.getLogger(__name__)


try:
    # optional: check Redis broker health
    cache.set("celery_health_check", "ok", timeout=5)

    # try to send task
    task = run_prediction_task.delay(...)
    logging.info(f"Submitted task ID: {task.id}")

except Exception as e:
    logging.error(f"Failed to dispatch Celery task: {e}")




def delete_file_after_delay(file_path, delay):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted after {delay} seconds.")
    threading.Timer(delay, delete_file).start()



def check_task_status(request, task_id):
    result = AsyncResult(str(task_id))
    if result.ready():
        if result.successful():
            return JsonResponse({
                'ready': True,
                'status': 'success',
                'file': f"{settings.MEDIA_URL}predictions_{task_id}.csv"
            })
        elif result.state == 'FAILURE':
            return JsonResponse({
                'ready': True,
                'status': 'error',
                'message': str(result.result)
            })
    return JsonResponse({'ready': False})

# РЈ▒№ИЈ Background prediction function
def predict(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            # چک کردن سایز فایل
            if uploaded_file.size > 2 * 1024 * 1024:
                form.add_error('file', 'The file size exceeds the limit of 2MB.')
                return render(request, 'predictor/upload.html', {'form': form})

            # ذخیره فایل آپلود شده
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            full_path = settings.MEDIA_ROOT / file_path

            try:
                new_df = pd.read_csv(full_path)

                # بررسی تعداد ردیف‌ها
                if len(new_df) > 10:
                    form.add_error('file', 'The uploaded CSV should not have more than 10 rows.')
                    return render(request, 'predictor/upload.html', {'form': form})

                # ساخت مسیر خروجی
                unique_filename = f'predictions_{uuid.uuid4().hex}.csv'
                results_file_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
                results_file_url = settings.MEDIA_URL + unique_filename

                # لاگ‌گیری و بررسی مقدارها
                if not results_file_path:
                    logger.error("results_file_path is empty or None.")
                    form.add_error(None, 'Internal error occurred. Please try again.')
                    return render(request, 'predictor/upload.html', {'form': form})

                if new_df.empty:
                    logger.error("Uploaded DataFrame is empty.")
                    form.add_error('file', 'The uploaded CSV is empty.')
                    return render(request, 'predictor/upload.html', {'form': form})

                try:
                    logger.info(f"Dispatching task with file: {results_file_path}")
                    run_prediction_task.delay(new_df.to_dict(orient='records'), results_file_path)
                except Exception as e:
                    logger.error(f"Failed to dispatch Celery task: {e}")
                    form.add_error(None, 'Failed to start prediction task. Please try again later.')
                    return render(request, 'predictor/upload.html', {'form': form})

                return render(request, 'predictor/results_pending.html', {
                    'results_file_url': results_file_url,
                    'message': 'Prediction is being processed. Please download after a few moments.'
                })

            except Exception as e:
                logger.exception(f"Error while handling uploaded file: {e}")
                form.add_error(None, 'Error reading your CSV file. Please check its format.')

            finally:
                if os.path.exists(full_path):
                    os.remove(full_path)

    else:
        form = CSVUploadForm()

    return render(request, 'predictor/upload.html', {'form': form})
    
    
    

# Other views
def redirect_to_predict(request):
    return redirect('predict', permanent=True)

def about(request):
    return render(request, 'predictor/about.html')

def contact(request):
    return render(request, 'predictor/contact.html')

def upload(request):
    return render(request, 'predictor/upload.html')