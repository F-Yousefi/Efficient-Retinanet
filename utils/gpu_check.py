import GPUtil
import time

def gpu_temperature(max, min):
    gpu = GPUtil.getGPUs()[0]
    if gpu.temperature > max:
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU temperature is beyond {gpu.temperature}c")
        print("Coolling down!")

        while gpu.temperature > min :
          time.sleep(5)
          gpu = GPUtil.getGPUs()[0]
          print(f"GPU temperature currently is {gpu.temperature}", end="\r")