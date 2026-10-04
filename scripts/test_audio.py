from noiseproof.data.grid import GRIDDataset
from noiseproof.preprocessing.audio import load_audio

dataset = GRIDDataset(
    root_dir="data/raw/grid"
)

sample = dataset[0]

# print(sample)

waveform, sample_rate = load_audio(sample["audio_path"])

info = {
    "speaker_id": sample["speaker_id"],
    "utterance_id": sample["utterance_id"],
    "transcript": sample["transcript"],
    "waveform_shape": waveform.shape,
    "sample_rate": sample_rate,
    "number_of_channels": waveform.shape[0],
    "number_of_audio_samples": waveform.shape[1],
    "duration": waveform.shape[1] / sample_rate,
    "waveform_min": waveform.min().item(),
    "waveform_max": waveform.max().item(),
}

print(info)