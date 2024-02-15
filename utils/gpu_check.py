import GPUtil
import time
from utils.config import config


def gpu_temperature():
    if config.gpu_under_control:
        time.sleep(config.delay_time)
        gpu = GPUtil.getGPUs()[0]
        if gpu.temperature > config.high_temp:
            gpu = GPUtil.getGPUs()[0]
            print(f"GPU temperature is beyond {gpu.temperature}c")
            print("Coolling down!")

            while gpu.temperature > config.low_temp:
                time.sleep(5)
                gpu = GPUtil.getGPUs()[0]
                print(f"GPU temperature currently is {gpu.temperature}", end="\r")
            print("\n")
