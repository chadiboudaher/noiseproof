from pathlib import Path

from noiseproof.data.grid import GRIDDataset


def create_fake_grid_sample(
    root: Path,
    speaker_id: str,
    utterance_id: str,
    alignment_text: str,
):
    audio_dir = root / "audio" / speaker_id
    video_dir = root / "video" / speaker_id
    alignment_dir = root / "alignments" / speaker_id

    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{utterance_id}.wav"
    video_path = video_dir / f"{utterance_id}.mpg"
    alignment_path = alignment_dir / f"{utterance_id}.align"

    audio_path.touch()
    video_path.touch()
    alignment_path.write_text(alignment_text)

    return audio_path, video_path, alignment_path


def test_grid_dataset_loads_matching_sample(tmp_path):
    create_fake_grid_sample(
        root=tmp_path,
        speaker_id="s1",
        utterance_id="bbaf2n",
        alignment_text=(
            "0 1000 sil\n"
            "1000 2000 bin\n"
            "2000 3000 blue\n"
            "3000 4000 at\n"
            "4000 5000 f\n"
            "5000 6000 two\n"
            "6000 7000 now\n"
            "7000 8000 sil\n"
        ),
    )

    dataset = GRIDDataset(tmp_path)

    assert len(dataset) == 1

    sample = dataset[0]

    assert sample["speaker_id"] == "s1"
    assert sample["utterance_id"] == "bbaf2n"
    assert sample["transcript"] == "bin blue at f two now"

    assert sample["audio_path"].name == "bbaf2n.wav"
    assert sample["video_path"].name == "bbaf2n.mpg"
    assert sample["alignment_path"].name == "bbaf2n.align"


def test_grid_dataset_skips_sample_with_missing_video(tmp_path):
    audio_dir = tmp_path / "audio" / "s1"
    alignment_dir = tmp_path / "alignments" / "s1"
    video_dir = tmp_path / "video" / "s1"

    audio_dir.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    (audio_dir / "bbaf2n.wav").touch()

    (alignment_dir / "bbaf2n.align").write_text(
        "0 1000 sil\n"
        "1000 2000 bin\n"
        "2000 3000 sil\n"
    )

    dataset = GRIDDataset(tmp_path)

    assert len(dataset) == 0


def test_grid_dataset_skips_sample_with_missing_alignment(tmp_path):
    audio_dir = tmp_path / "audio" / "s1"
    video_dir = tmp_path / "video" / "s1"
    alignment_dir = tmp_path / "alignments" / "s1"

    audio_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)

    (audio_dir / "bbaf2n.wav").touch()
    (video_dir / "bbaf2n.mpg").touch()

    dataset = GRIDDataset(tmp_path)

    assert len(dataset) == 0