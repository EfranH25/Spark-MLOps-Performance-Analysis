import kagglehub
import threading
import time
import psutil
from typing import List, Dict
from os import path, mkdir, rename, listdir

import logging

logger = logging.getLogger("ml_app")


class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.is_running = False
        self.measurements: List[Dict] = []
        self.peak_cpu = 0
        self.peak_memory = 0
        self._monitor_thread = None

    def _monitor(self):
        process = psutil.Process()
        while self.is_running:
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            current_memory_mb = memory_info.rss / (1024 * 1024)

            # Update peak values
            self.peak_cpu = max(self.peak_cpu, cpu_percent)
            self.peak_memory = max(self.peak_memory, current_memory_mb)

            measurement = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_mb": current_memory_mb,
            }
            self.measurements.append(measurement)
            time.sleep(self.interval)

    def stop(self):
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def start(self):
        self.is_running = True
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.daemon = True  # Thread will exit when main program exits
        self._monitor_thread.start()

    def get_stats(self) -> Dict:
        return {
            "peak_cpu_percent": round(self.peak_cpu, 2),
            "peak_memory_mb": round(self.peak_memory, 2),
            "measurements": self.measurements,
        }


def setup_folders():
    if not path.exists("data"):
        mkdir("data")

    if not path.exists("metrics"):
        mkdir("metrics")

    if not path.exists("models"):
        mkdir("models")


def get_data(data_handle: str, data_folder: str) -> str | None:
    data_output_folder = path.join("data", data_folder)
    if path.exists(data_output_folder) and listdir(data_output_folder):
        logger.info(
            f"Files in output path '{data_output_folder}':\n{listdir(data_output_folder)}\n"
            f"Please clean if unintended."
        )
        return None
    else:
        # TODO: for some reason kaggle download doesn't save to path so have to manually move
        data_path = kagglehub.dataset_download(handle=data_handle)
        rename(data_path, data_output_folder)
        logger.info(f"Dataset {data_handle} downloaded to {data_output_folder}")
        return data_output_folder
