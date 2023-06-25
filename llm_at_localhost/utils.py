from pynvml import *

import psutil
import torch

def get_mem_utilization(device=0):
    """
    Return memory usage in GB
    """
    if torch.has_cuda:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used/1024**3
    else:
        memory = psutil.virtual_memory()
        return memory.used / (1024 ** 3)