import torch

def self_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available:
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
        running_instance = 'Train_1'
    return device 