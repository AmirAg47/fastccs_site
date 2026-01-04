import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
from .utils import (
    load_encoder, load_model, load_scaler, load_cluster, predict_data,
    safe_compute_properties, property_columns
)
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
import logging
import pandas as pd
from .utils import (
    load_encoder, load_model, load_scaler, load_cluster, predict_data,
    safe_compute_properties, property_columns
)

logger = logging.getLogger(__name__)
 
# @shared_task
# def delete_file_later(file_path):
#     try:
#         logging.info(f"Task to delete file {file_path} scheduled.")
#         time.sleep(50)  # Simulate 5-minute delay

#         logging.info(f"Attempting to delete file {file_path} after 5-minute delay.")
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             logging.info(f"File {file_path} deleted after delay.")
#         else:
#             logging.warning(f"File {file_path} does not exist.")
#     except Exception as e:
#         logging.error(f"Failed to delete file {file_path}: {e}")
encoder = load_encoder()
interpreter = load_model()
scaler = load_scaler()
kmeans = load_cluster()

@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def run_prediction_task(self, data_dict, results_file_path):
    try:
        logger.info(f"Prediction task started with file path: {results_file_path}")
        
        # تبدیل داده‌های دریافتی به DataFrame
        new_df = pd.DataFrame(data_dict)
        logger.info(f"Input data shape: {new_df.shape}")

        # تبدیل ستون 'Adduct' به one-hot encoding
        new_adducts = new_df["Adduct"].values.reshape(-1, 1)
        one_hot = encoder.transform(new_adducts)
        one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(["Adduct"]))
        df_encoded = pd.concat([new_df, one_hot_df], axis=1)
        
        # محاسبه ویژگی ها
        logger.info("Computing molecular properties...")
        property_matrix = df_encoded.apply(
            lambda row: safe_compute_properties(row["Smiles"], row["Adduct"]),
            axis=1,
            result_type="expand"
        )
        
        # اختصاص نام ستون‌ها به ماتریس ویژگی‌ها
        property_matrix.columns = property_columns
        
        # اتصال یک‌باره به DataFrame اصلی
        df_encoded = pd.concat([df_encoded, property_matrix], axis=1)
    
        
        logger.info("Starting prediction...")
        # اجرای مدل پیش‌بینی
        predictions_df = predict_data(interpreter, df_encoded, scaler, kmeans)

        # ذخیره نتایج در فایل CSV
        try:
            predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
            logger.info(f"Prediction saved successfully to: {results_file_path}")
        except Exception as save_error:
            logger.error(f"Failed to save CSV: {save_error}", exc_info=True)
            raise save_error  # تا تسک بداند خطا رخ داده

    except Exception as exc:
        logger.error(f"Celery prediction task failed: {exc}", exc_info=True)
        # تلاش مجدد تسک با تأخیر ۵ ثانیه، حداکثر ۳ بار
        raise self.retry(exc=exc, countdown=5, max_retries=3)
        
        
