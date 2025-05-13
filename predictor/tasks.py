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

 
@shared_task
@shared_task
def delete_file_later(file_path):
    try:
        logging.info(f"Task to delete file {file_path} scheduled.")
        time.sleep(50)  # Simulate 5-minute delay

        logging.info(f"Attempting to delete file {file_path} after 5-minute delay.")
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File {file_path} deleted after delay.")
        else:
            logging.warning(f"File {file_path} does not exist.")
    except Exception as e:
        logging.error(f"Failed to delete file {file_path}: {e}")

@shared_task
def run_prediction_task(data_dict, results_file_path):
    try:
        logging.info("Prediction task started.")
        new_df = pd.DataFrame(data_dict)

        encoder = load_encoder()
        new_adducts = new_df["Adduct"].values.reshape(-1, 1)
        one_hot = encoder.transform(new_adducts)
        one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(["Adduct"]))
        df_encoded = pd.concat([new_df, one_hot_df], axis=1)

        properties_df = df_encoded.apply(
            lambda row: pd.Series(safe_compute_properties(row['Smiles'], row['Adduct'])),
            axis=1
        )
        properties_df.columns = property_columns
        df_encoded = pd.concat([df_encoded, properties_df], axis=1)

        interpreter = load_model()
        scaler = load_scaler()
        kmeans = load_cluster()

        predictions_df = predict_data(interpreter, df_encoded, scaler, kmeans)
        predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')

        logging.info(f"Prediction saved to: {results_file_path}")
        # delete_file_later.apply_async(args=[results_file_path], countdown=50)  # 300 seconds = 5 minutes

    except Exception as e:
        logging.error(f"Celery prediction task failed: {e}")
