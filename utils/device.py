import torch
""" 
This file is used to get the device of the running instance.
    input : None
    output : device : str : the device of the running instance.
    mps for METAL GPU APPLE SILICON
    cuda for NVIDIA GPU
    cpu for CPU if no GPU is available
"""
def self_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available:
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")

    return device 