from pathlib import Path

import torch
import torchaudio
from torchaudio import functional as f

TARGET_SAMPLE_RATE = 16_000

class AudioException(Exception):
    pass

def load_audio(audio_path: Path):
    if not audio_path.exists():
        raise AudioException(
            f"Audio path does not exist: {audio_path}"
        )

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        raise AudioException(
            f"Failed to load {audio_path}: {e}"
        ) from e

    # Convert multi-channel audio to mono (Already tested to mono)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if sample rate is not 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = f.resample(
            waveform=waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE

    return waveform, sample_rate