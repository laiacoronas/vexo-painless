# AST trained on TAME

Frozen Audio Spectrogram Transformer (AST), with general and reference-conditioned pain regression heads trained on TAME Pain 1.0.0.

## Model files

| Component | File | Role |
|---|---|---|
| AST base | [`base/model.safetensors`](base/model.safetensors) | AudioSet-pretrained encoder, frozen during TAME training; stored with Git LFS |
| General head | [`heads/general.pt`](heads/general.pt) | Predicts pain from one query phrase |
| Reference head | [`heads/personal.pt`](heads/personal.pt) | Predicts pain using a query and three earlier rated phrases from the same person and session |

The base is [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), revision `f826b80d28226b62986cc218e5cec390b1096902`. Its configuration, feature extractor configuration, upstream model card and BSD-3-Clause license are included in `base/`. The trained TAME weights live in the heads; the AST encoder retains its pretrained weights.

Audio is averaged to mono, resampled to 16 kHz and converted with `ASTFeatureExtractor` to 128 mel bins and 1,024 time frames. The bundled extractor uses mean −4.2677393 and standard deviation 4.5689974. `ASTModel.pooler_output` supplies a 768-dimensional embedding. Embedding normalization uses unique training clips only; the saved mean and standard deviation are included in each head checkpoint.

The general head is `768 → 128 → 64 → 1`. The reference head is `2305 → 128 → 64 → 1`, using the query embedding, reference embedding, their difference, and the reference rating divided by 10. Both use GELU after the first two linear layers and dropout 0.2 after the first GELU. For each reference, the reference head predicts a correction to its rating; the final prediction averages the three corrected ratings.

## Dataset and division

[TAME Pain 1.0.0](https://physionet.org/content/tame-pain/1.0.0/) contains 7,044 audio files from 51 participants. This experiment uses the `mic1_trim_v1` audio archive and `meta_audio.csv`.

| Division | Participants | Query phrases |
|---|---:|---:|
| Training | 35 | 1,188 |
| Validation | 8 | 356 |
| Test | 8 | 303 |

Participants are sorted by ID, shuffled with Python's random seed 42, and assigned in the counts above. Each participant belongs to one division. [`prepare.py`](prepare.py) regenerates the exact split from the source metadata; [`results/split_protocol.json`](results/split_protocol.json) records the seed, counts, selection policy, metadata hash and expected manifest hash.

Eligible phrases have a finite revised pain rating from 1 to 10 and `ACTION LABEL == 0`. Spoken rating clips (`UTTID == 99999`), copied or missing ratings, and phrases whose rating disagrees with the following spoken report are excluded. Within each participant/condition session, utterance order defines chronology. The first usable phrase from each of the first three usable rating blocks becomes a reference. Every phrase in those three blocks is reserved; later blocks supply queries. Both heads are evaluated on the same queries. References can have repeated pain ratings.

Authorized dataset access is available through PhysioNet under its [Restricted Health Data License](https://physionet.org/content/tame-pain/view-license/1.0.0/). The repository provides model weights, aggregate results and split-generation code. Obtain the source audio and metadata through PhysioNet to reproduce the experiment.

## Training

Only the regression heads are optimized. Training uses CPU float32, seed 42, AdamW with learning rate 0.001 and weight decay 0.001, batch size 64, Huber loss with delta 1, and gradient norm clipping at 1. Each query is weighted inversely to its participant's query count, with weights normalized to mean 1.

The maximum is 100 epochs, with early stopping after 10 epochs without improvement. Checkpoint selection uses validation participant-macro MAE: compute MAE for each participant, then average equally across participants. The selected general head is epoch 31; the selected reference head is epoch 2. [`results/environment.json`](results/environment.json) preserves the original training environment, and the two history files contain every logged epoch.

The exported training script loads the bundled, pinned base locally. Its architecture, fitting procedure and evaluation match the original run. [`results/provenance.json`](results/provenance.json) records original/exported source hashes and checkpoint metadata. [`SHA256SUMS`](SHA256SUMS) records package file hashes.

## Results

MAE is measured in pain-rating points. Validation selects the learned checkpoints; the test set contains eight held-out participants and 303 phrases.

| Model or baseline | Selected epoch | Validation participant-macro MAE | Test participant-macro MAE |
|---|---:|---:|---:|
| AST + general head | 31 | 1.857 | 2.151 |
| AST + reference head | 2 | 0.631 | 0.706 |
| Mean of three reference ratings | — | — | 0.536 |
| Latest reference rating | — | — | **0.453** |
| Training participant-mean rating | — | — | 2.095 |

The reference head improves over the general head by 1.445 MAE points; the paired participant bootstrap 95% interval is [1.019, 1.860], from 10,000 resamples. Both reference-rating baselines outperform the learned reference head, which limits the evidence for added predictive value from audio in this split.

Predictions are raw, unbounded outputs. The general and reference heads produce 21 and 52 predictions outside the training scale, respectively. The test queries contain 208 low-pain phrases (1–3), 56 moderate-pain phrases (4–6), and 39 high-pain phrases (7–10). [`results/test_summary.json`](results/test_summary.json) includes exact aggregate metrics and error by pain band. Pain-band MAE averages phrases within each band; the main score averages participants.

These results describe one split of this controlled recording protocol. Eight test participants and repeated phrases within sessions limit precision and generalization to new recording conditions.

## Run inference

From the repository root:

```bash
git lfs install
git lfs pull --include="models/ast_tame/base/model.safetensors"
cd models/ast_tame
shasum -a 256 -c SHA256SUMS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py query.wav --head general
python predict.py query.wav --head personal \
  --reference reference_1.wav 2 \
  --reference reference_2.wav 4 \
  --reference reference_3.wav 5
```

Supply one already segmented spoken phrase per WAV. For the reference head, use three earlier phrases with known ratings from the same person and session, following the policy above. The CLI returns a raw prediction and whether it falls outside the 1–10 training scale. It loads model files locally and defaults to CPU float32.

## Reproduce training and evaluation

Run from `models/ast_tame` after obtaining the dataset and installing the dependencies. Place the metadata at `data/tame/meta_audio.csv` and extracted audio under `data/tame/mic1_trim_v1/` (searched recursively).

```bash
python -m unittest test_prepare test_model
python prepare.py --metadata data/tame/meta_audio.csv --output runs/reproduction/manifest.json
python train.py extract --device cpu \
  --manifest runs/reproduction/manifest.json \
  --audio data/tame/mic1_trim_v1 \
  --cache runs/reproduction/embeddings.pt --output runs/reproduction
python train.py fit --device cpu \
  --manifest runs/reproduction/manifest.json \
  --cache runs/reproduction/embeddings.pt --output runs/reproduction
python train.py evaluate --device cpu \
  --manifest runs/reproduction/manifest.json \
  --cache runs/reproduction/embeddings.pt --output runs/reproduction
```

The regenerated manifest should match the hash in `results/split_protocol.json`. Use a fresh output directory for another run. Exact numerical reproducibility can depend on platform and library versions; the original versions are pinned in `requirements.txt`.

## Sources

- Dao et al. (2025), *TAME Pain: Trustworthy AssessMEnt of Pain from Speech and Audio for the Empowerment of Patients*, PhysioNet, version 1.0.0. [DOI: 10.13026/20e2-1g10](https://doi.org/10.13026/20e2-1g10).
- Gong, Chung and Glass (2021), *AST: Audio Spectrogram Transformer*. [Paper](https://arxiv.org/abs/2104.01778) · [original code](https://github.com/YuanGongND/ast).
- Pollard et al. (2026), *PhysioNet as a global platform for biomedical research*. [DOI: 10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).
