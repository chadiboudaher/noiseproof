content = """# NoiseProof: Noise-Robust Audio-Visual Speech Recognition with Learned Modality Gating

**Author:** Chadi Boudaher  
**Status:** Proposal  
**Duration:** 5 weeks

## 1. Summary

NoiseProof is an audio-visual speech recognition (AVSR) system that combines speech audio with lip video and learns how much to trust each stream as acoustic noise changes. The project measures how much the visual channel improves recognition across noise levels and serves the model through a FastAPI endpoint.

## 2. Problem Statement

Audio-only speech recognition degrades sharply in noisy environments such as factories, hospitals, vehicles, and crowded public spaces. Lip movements are unaffected by acoustic noise, but most deployed systems ignore them. The gap is wider for low-resource dialects such as Lebanese Arabic, where little training data exists and noise robustness cannot be bought with more data.

## 3. Research Question

**At what noise level does adding vision start to improve recognition, and does learned, noise-aware fusion outperform fixed fusion?**

### Hypotheses

- **H1:** Audio-only WER rises sharply as SNR falls, while video-only WER stays roughly constant.
- **H2:** Fusion beats audio-only below a crossover SNR, and the gap widens as noise increases.
- **H3:** Learned gating outperforms fixed concatenation fusion at low SNR, and its audio weight decreases as noise increases.

## 4. Objectives

1. Build a reproducible pipeline for data loading, lip-region extraction, and controlled noise mixing at exact SNRs.
2. Train four systems under identical conditions:
   - Audio-only
   - Video-only
   - Concatenation fusion
   - Gated fusion
3. Evaluate all four on held-out speakers across SNR levels and report WER and CER.
4. Analyze the learned modality weights as a function of SNR.
5. Serve the best model through a FastAPI endpoint that returns a transcript and modality-weight trace.

## 5. Approach

| Component      | Design                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------- |
| Audio encoder  | `wav2vec2-base`, frozen initially                                                        |
| Visual encoder | 3D-conv stem + ResNet-18, trained from scratch                                           |
| Temporal model | 2-layer BiGRU (Conformer if time allows)                                                 |
| Fusion         | (a) Concatenation, (b) per-timestep sigmoid gate over projected audio and video features |
| Decoder        | Linear layer with character-level CTC                                                    |

### Key Training Choices

- Frame-rate alignment between video (~25 fps) and audio features (~50 Hz) before fusion.
- Random-SNR noise augmentation during training, including clean audio.
- Modality dropout, so the model cannot ignore either stream.

## 6. Data

- **Speech:** GRID corpus, split by speaker so train, validation, and test speakers are disjoint. Speaker-held-out evaluation avoids identity leakage. Test speakers are fixed before any results are seen.
- **Noise:** MUSAN or DEMAND, mixed at clean, 10, 5, 0, and -5 dB SNR, with fixed seeds and fixed noise files for the test set. Licenses will be checked before use.
- **Future extension:** Fine-tuning on the Lebanese Arabic corpus being built for the thesis.

## 7. Evaluation

- **Metrics:** WER and CER on held-out speakers, ideally averaged over 3 seeds.
- **Headline figure:** WER vs. SNR for all four systems.
- **Analysis figures:** Mean audio-vs-video gate weight vs. SNR, plus example utterances where fusion corrects audio-only errors.
- **Ablations:** Fixed vs. learned fusion, with and without modality dropout.

## 8. Limitations

These limitations will be stated explicitly in the README:

- GRID has a small vocabulary and fixed grammar, so absolute WERs will look unrealistically low. The contribution is the shape of the WER-vs-SNR curves, not the clean-audio number.
- The audio encoder is pretrained and the visual encoder is trained from scratch, which favors audio. This will be stated plainly in the results.
- Noise is synthetically mixed rather than recorded in real environments.

## 9. Deliverables

- Public GitHub repository with code, configs, tests, and a README with results.
- Result CSVs and figures:
  - WER vs. SNR
  - Gate weight vs. SNR
- FastAPI service with a `POST /transcribe` endpoint and Dockerfile.
- Integration as a job type in the ML Media Orchestrator.

## 10. Timeline

| Week | Milestone                                                                                                                                      |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Repo setup, GRID download, speaker splits, lip crops, noise mixer with tests, audio-only baseline and noise sweep, video-only training started |
| 2    | Video-only complete, concatenation fusion with modality dropout                                                                                |
| 3    | Gated fusion, ablations, multiple seeds                                                                                                        |
| 4    | Plots, error analysis, README with results                                                                                                     |
| 5    | FastAPI endpoint, Dockerfile, Orchestrator integration                                                                                         |

## 11. Risks and Mitigations

| Risk                                 | Mitigation                                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------------------- |
| Video-only WER is poor on small data | Report it honestly. Fusion can still win at low SNR.                                          |
| Gate collapses to audio              | Increase modality dropout and low-SNR sampling.                                               |
| Compute is limited                   | Freeze wav2vec2, shrink crops, and reduce epochs.                                             |
| Time runs short                      | A finished audio-only vs. video-only vs. concatenation comparison is still a complete result. |

## 12. Relation to the Thesis

The pipeline—lip-ROI extraction, CTC training, WER/CER evaluation, and noise-robustness analysis—transfers directly to visual speech recognition for Lebanese Arabic.

GRID results are reported separately and are never presented as Lebanese Arabic results. Once the Lebanese corpus is ready, it can be used for future fine-tuning and evaluation.
"""

path = "/mnt/data/PROPOSAL.md"
with open(path, "w", encoding="utf-8") as f:
f.write(content)

print(path)
