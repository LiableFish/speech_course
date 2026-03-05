import torch
import torch.nn as nn
import torchaudio
import numpy as np


class PitchShiftResample(nn.Module):
    def __init__(self, sample_rate: int, n_semitones: tuple[int, int]) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_semitones, self.max_semitorns = n_semitones

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        n_semitones = np.random.randint(self.min_semitones, self.max_semitorns + 1)
        factor = 2.0 ** (n_semitones / 12.0)
        target_freq = int(self.sample_rate * factor)

        return torchaudio.functional.resample(
            waveform, 
            orig_freq=target_freq, 
            new_freq=self.sample_rate
        )
    
class Power(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x).pow(2)
    

class RandomTimeStretch(torchaudio.transforms.TimeStretch):
    def __init__(self, hop_length = None, n_freq = 201, fixed_rate = None, *, rate: tuple[float, float]):
        super().__init__(hop_length, n_freq, None)
        self.min_rate, self.max_rate = rate

    def forward(self, complex_specgrams, overriding_rate = None):
        overriding_rate = np.random.uniform(self.min_rate, self.max_rate)
        return super().forward(complex_specgrams, overriding_rate)
    
    