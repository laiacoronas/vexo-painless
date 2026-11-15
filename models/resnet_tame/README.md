# ResNet pain model and personal heads

TAME-trained ResNet-34, with width-16 personal heads for Milan and Lamita. The personal reference heads compare each query phrase with three pain-0 reference embeddings saved in the checkpoint.

## Model files

| Component | File | Architecture |
|---|---|---|
| Shared ResNet and TAME general head | [base/pain_model.pt](base/pain_model.pt) | ResNet-34 → 512 → 128 → 64 → 1 |
| Milan reference head | [heads/milan.pt](heads/milan.pt) | 1536 → 16 → 1 |
| Lamita reference head | [heads/lamita.pt](heads/lamita.pt) | 1536 → 16 → 1 |
| Milan audio-only comparison | [heads/milan-audio.pt](heads/milan-audio.pt) | 512 → 16 → 1 |
| Lamita audio-only comparison | [heads/lamita-audio.pt](heads/lamita-audio.pt) | 512 → 16 → 1 |

The shared model is stored with Git LFS. Each personal checkpoint contains its learned weights, embedding mean and standard deviation, target mean, selected settings, encoder checksum, and reference embeddings where applicable. [model_index.json](model_index.json) maps every head to its results and source checksums. [SHA256SUMS](SHA256SUMS) covers the package files.

## Run predictions

From the repository root:

```bash
git lfs install
git lfs pull --include="models/resnet_tame/base/pain_model.pt"
cd models/resnet_tame
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements.txt
python predict.py phrase.wav --head milan
python predict.py phrase.wav --head lamita
python predict.py phrase.wav --head general
```

Use one segmented spoken phrase per input. `milan` and `lamita` use the three saved pain-0 references for that person. `milan-audio` and `lamita-audio` run the corresponding audio-only comparison. Inference uses local files and CPU float32. The CLI verifies checkpoint checksums and reports the raw prediction, audio duration, frame count, and reference count. Outputs can extend beyond the reported 0–10 scale.

## Audio processing

Audio is averaged to mono and resampled to 16 kHz with polyphase resampling. Each phrase is peak-normalized. A 128-bin mel spectrogram uses a 400-sample FFT, 160-sample hop and 50–8,000 Hz range. The pipeline applies `log(mel + 1e-6)` and per-phrase min-max normalization. It keeps the first 300 frames and right-pads shorter inputs with zeros. The single-channel ResNet-34 produces a 512-dimensional pooled embedding.

The exact preprocessing and architecture are preserved in [training/resnet_experiment.py](training/resnet_experiment.py). The original AVPENet architecture reference and its MIT license are included in the training sources.

## TAME training

The backbone starts from torchvision's ImageNet-1K V1 ResNet-34 weights. Its first convolution is adapted to one channel by averaging the RGB weights. Two variants were compared: a frozen encoder with a new regression head, and training the final residual group (`layer4`) together with that head. Batch-normalization running statistics stay fixed. Both use a `512 → 128 → 64 → 1` head with GELU and dropout 0.2.

The seed-42 participant split contains 35 training participants / 1,188 queries, 8 validation participants / 356 queries, and 8 test participants / 303 queries. Eligible TAME phrases have quality flag 0 and a valid pain rating from 1 to 10 matching the following spoken report. Each session's first three usable rating blocks are reserved for references; later blocks supply queries. Participants remain in a single split. The selection and split-generation code is [training/prepare.py](training/prepare.py); [results/tame/split_protocol.json](results/tame/split_protocol.json) preserves the metadata and manifest checksums, filtering audit, and counts.

Training uses participant-weighted Huber loss, delta 1, AdamW, batch size 32, head learning rate 0.001, final-group learning rate 0.00001, and weight decay 0.001. The maximum is 100 epochs with patience 10. Validation participant-macro MAE selects the model: the final-group variant at epoch 3. Embedding normalization is fitted on training queries. The run used an Apple M4 Pro; [environment.json](results/tame/environment.json) records package versions and hardware.

| Predictor | Test participant-macro MAE |
|---|---:|
| Frozen ResNet + head | 2.185 |
| Selected final-group ResNet + head | 2.076 |
| Training participant-mean rating | 2.095 |
| Training median rating | 1.647 |
| Mean of personal reference ratings | 0.536 |
| Latest personal reference rating | 0.453 |

The selected model improves MAE by 0.109 over the frozen variant; the paired participant-bootstrap 95% interval is −0.170 to +0.432. The test set has 208 low-pain, 56 moderate-pain and 39 high-pain phrases. The constant median and reference-rating controls outperform the learned model on this split. Full aggregate results, histories, configuration, and measured latency are in [results/tame](results/tame).

TAME source audio and metadata are obtained through [PhysioNet](https://physionet.org/content/tame-pain/1.0.0/) under its access terms. This package provides aggregate TAME results and split-generation code.

## Personal training for Milan and Lamita

The selected TAME ResNet is frozen. New personal heads are fitted to each person's labeled repetitions of “vexohacks is cool.” Ratings follow each group in the source recording. All three pain-0 takes enter calibration. At each other pain level, the first two takes enter calibration and the remaining takes enter evaluation.

| Person | Pain levels | Calibration phrases | Evaluation phrases |
|---|---|---:|---:|
| Milan | 0, 2, 5, 6, 7, 8, 9, 10 | 17 | 10 |
| Lamita | 0, 4, 6, 10 | 9 | 6 |

For the reference head, the input for each query-reference pair is the normalized query embedding, normalized reference embedding, and their difference: `512 + 512 + 512 = 1536` features. A `1536 → 16 → 1` MLP with GELU predicts a centered query score. Predictions are averaged over the available references and the training-target mean is added. A query matching a reference excludes its self-pair. Each query contributes equally to MSE after reference averaging. The audio-only head uses the same query embeddings and labels with `512 → 16 → 1`.

Both variants use full-batch AdamW, learning rate 0.001, seed 42, and candidate weight decays 0.01, 0.1, and 1. Every epoch from 1 through 400 is checked. Two calibration folds train on take 1 and validate on take 2, then swap; extra calibration takes, including the third zero-pain take, stay in training. Within each fold, scaling and target centering use training phrases, and the reference pool uses its pain-0 training phrases. Each reference fold therefore has two pain-0 references; the final fit has three.

Mean calibration-fold MAE selects weight decay and epoch, with ties resolved by stronger decay and then earlier epoch. The selected model is refitted on all calibration phrases and saved before evaluation. The per-epoch evaluation curves are retrospective diagnostics; the reported checkpoints use calibration selection.

| Person | Head | Selected epoch | Weight decay | Calibration CV MAE | Evaluation MAE | Evaluation RMSE |
|---|---|---:|---:|---:|---:|---:|
| Milan | Audio-only | 10 | 1.0 | 2.251 | 1.296 | 1.618 |
| Milan | Three pain-0 references | 6 | 1.0 | 1.985 | **0.991** | 1.244 |
| Lamita | Audio-only | 400 | 1.0 | 2.347 | 1.011 | 1.169 |
| Lamita | Three pain-0 references | 4 | 1.0 | 2.284 | **0.946** | 1.069 |

Reference inputs reduce matched evaluation MAE by 0.305 for Milan and 0.065 for Lamita. Both comparisons use identical calibration/evaluation membership within each person. These are exploratory results on reused, within-recording splits. Calibration and evaluation phrases share rating blocks; the small evaluation sets measure performance on nearby repetitions from these sessions. The reference head also has more parameters: 24,609 versus 8,225 for the audio-only head.

- [Milan results, curves, selected settings, and split](results/milan)
- [Lamita results, curves, selected settings, and split](results/lamita)
- [Export verification](results/export_verification.json)

## Reproduce personal training

The original source phrases remain with the recording owner. Supply the directory containing their `recordings/` subdirectory. The saved plan fixes membership, ratings and settings; source audio checksums are verified before training.

```bash
python train_person.py --person milan --audio-root /path/to/milan_pain --output runs/milan-reproduction
python train_person.py --person lamita --audio-root /path/to/lamita_pain --output runs/lamita-reproduction
```

Each command extracts frozen ResNet embeddings, tunes and trains the matched audio-only and reference heads, and saves every-epoch calibration/evaluation curves, selected checkpoints, and per-phrase results. Use a fresh output directory for each run. Numerical results can vary with platform and library versions; `requirements.txt` pins the original ResNet environment.

## Reproduce TAME training

After obtaining the source dataset, run from this package directory:

```bash
python -m unittest discover -s training -p 'test_resnet_experiment.py'
python training/prepare.py --metadata data/tame/meta_audio.csv --output runs/reproduction/manifest.json
python training/resnet_experiment.py all --device cpu \
  --manifest runs/reproduction/manifest.json \
  --audio data/tame/mic1_trim_v1 --output runs/reproduction/resnet
```

The initial torchvision ImageNet checkpoint is downloaded when its local cache is empty. The prepared split hash should match [results/tame/split_protocol.json](results/tame/split_protocol.json). Original personal-training orchestration and helper sources are preserved in `training/`; `train_person.py` adapts their paths to this package.

## Sources

- [TAME Pain 1.0.0](https://physionet.org/content/tame-pain/1.0.0/)
- [AVPENet audio encoder](https://github.com/oussama123-ai/AVPENet/blob/main/avpenet/models/audio_encoder.py)
- [torchvision ResNet-34](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)
