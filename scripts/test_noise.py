from pathlib import Path
from noiseproof.data.grid import GRIDDataset
from noiseproof.preprocessing.audio import load_audio
from noiseproof.preprocessing.noise import (
    calculate_power,
    match_noise_length,
    calculate_noise_scale
)

dataset = GRIDDataset(
    root_dir="data/raw/grid"
)

sample = dataset[0]

waveform_speech, sample_rate_speech = load_audio(sample["audio_path"])
waveform_noise, sample_rate_noise = load_audio(
    Path(
        r"C:\Users\chadi\OneDrive\Desktop\fastapi-l\noiseproof\data\raw\noise\ch14.wav"
    )
)

waveform_noise = match_noise_length(
    waveform_noise,
    waveform_speech.shape[1]
)

speech_power = calculate_power(waveform_speech)
noise_power = calculate_power(waveform_noise)

# print(f"Speech shape: {waveform_speech.shape}")
# duration_speech = waveform_speech.shape[1] / sample_rate_speech
# print(f"Speech duration: {duration_speech}")
# print(f"Speech power: {speech_power.item():.4f}")

# print(f"Noise shape: {waveform_noise.shape}")
# duration_noise = waveform_noise.shape[1] / sample_rate_noise
# print(f"Noise duration: {duration_noise}")
# print(f"Noise power: {noise_power.item():.8f}")

print("-"*20)

print(f"At -10dB: {calculate_noise_scale(speech_power, noise_power, -10)}")
print(f"At -5dB: {calculate_noise_scale(speech_power, noise_power, -5)}")
print(f"At 0dB: {calculate_noise_scale(speech_power, noise_power, 0)}")
print(f"At 5dB: {calculate_noise_scale(speech_power, noise_power, 5)}")
print(f"At 10dB: {calculate_noise_scale(speech_power, noise_power, 10)}")
