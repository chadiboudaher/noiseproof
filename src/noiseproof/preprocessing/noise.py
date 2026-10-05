import torch

def calculate_power(waveform: torch.Tensor):
    signal_power = torch.mean(waveform ** 2)

    return signal_power