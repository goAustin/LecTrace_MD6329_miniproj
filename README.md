# Do numbers survive the trip into the model?

A [LecTrace](https://praisegee.github.io/lectrace/) lecture for the 10-minute *Execution-Explanation Presentation*.
Mini-project: ICU mortality prediction on PhysioNet 2012. A 0.29M-parameter event encoder trained from scratch
(`no_llm`) against Llemr-v1, a 7.1B-parameter EHR language model fine-tuned as in its paper.

- Lecture: [`01_icu_mortality.py`](01_icu_mortality.py) (47 steps, about 10 minutes at the planned pace)
- Demo Link (after you deploy, see below): `https://<your-github-user>.github.io/<your-repo>/`

## Structure of the talk

| Part | Steps | What the audience sees | Time |
|---|---|---|---|
| Problem, motivation, approach | 2-6 | ICU mortality prediction, Llemr vs. a small encoder, cost table | 2:13 |
| State 1 | 7-14 | a chart row becomes `(z, t_abs, dt)` | 1:28 |
| State 2 | 15-22 | the event becomes one 128-d token (variable + value + time embeddings) | 1:25 |
| State 3 | 23-30 | 392 tokens, 2-layer Transformer, mean pooling, logit, risk | 1:21 |
| State 4 | 31-39 | what if the lactate were 4 mmol/L? patient, population, text-encoder cosines | 1:55 |
| Key result | 40-47 | AUROC, paired bootstrap, value shuffle, healthcare interpretation, limits | 1:39 |

One real test patient (ID 139655: 72-year-old man, cardiac surgery recovery unit, survived) is followed through the
small model. The lecture never shows the whole trace: only the four states above.

## Run it

Python 3.11 or newer.

```bash
pip install lectrace numpy matplotlib
lectrace serve            # http://localhost:7000, reloads when you save (use --port to change)
lectrace run 01_icu_mortality.py     # prints the step count and any error
```

## Publish it (the "Demo Link")

1. Create a GitHub repository and push this folder to its `main` branch.
2. In the repository: Settings, Pages, Source: **GitHub Actions**.
3. The workflow in `.github/workflows/lectrace.yml` builds the lecture (`lectrace build --output _site`, with the
   packages of `requirements.txt`) and deploys it. The page appears at `https://<user>.github.io/<repo>/`.
4. Put that link under "Demo Link" on the cover page of the report.

`lectrace build --output _site` also works locally; `_site/` is ignored by git.

## What is computed live, what is recorded

| Shown in the lecture | Status |
|---|---|
| everything the small model does to patient 139655: z-score, time features, the three embeddings, the token, the 2-layer Transformer, the mean, the logit, the risk, the lactate what-if | **live**: numpy with the trained `no_llm` weights |
| AUROC of both models, the paired bootstrap, Llemr's AUROC with shuffled values | **live**: computed from the saved test predictions (`data/preds`, `data/llemr_diagnostics`) |
| Llemr's input text for an event (`Blood Lactate Blood Gas: 1.3 mmol/L`) | **live**: the formatting rule of the project (checked against the 9 recorded probe texts) |
| Llemr's risk for patient 139655 and its mean risk over 633 patients per lactate value | **recorded**: Llemr (7.1B) cannot run in a browser, its saved outputs are read from `data/` |
| `no_llm` mean risk over 633 patients per lactate value, `no_llm` AUROC with shuffled values | **recorded**: project runs E3 and E2 (`data/results`) |
| AUPRC, training times, parameter counts, GPU memory, cosines of the text encoder | **recorded**: copied from the result files (`data/recorded_numbers.json`, `data/numeric_sensitivity_excerpt.json`) |

## Where each data file comes from

All paths refer to the project `dl_proj/`. Nothing is simulated or typed in by hand except `recorded_numbers.json`,
whose blocks name their source files.

| File in `data/` | Origin |
|---|---|
| `no_llm_s1_weights.npz` | `checkpoints/no_llm_s1.pt` (seed 1, split 1; SHA-256 `e424f216...022f`) converted by `tools/export_weights.py`, which reads the checkpoint without PyTorch (42 tensors, 288,961 parameters) |
| `patient_139655/event_tokens.npz`, `observations_raw.csv` | `demo/_cache/` (the model input of the demo patient and its raw record lines) |
| `patient_139655/patient.json` | `demo/_cache/patient.json` (one line added: `_source`) |
| `patient_139655/recorded_no_llm.json` | the `no_llm` entries of `demo/_cache/risks.json` (CPU, bf16 autocast) |
| `stats_split1_lactate.json` | the Lactate row of `artifacts/stats_split1.json` (training-set clip bounds, mean, std) |
| `variable_map.csv` | `results/llemr/variable_map.csv` (how P12 variables are written for Llemr) |
| `preds/no_llm_s1.csv`, `preds/llemr_ft_s1.csv` | `results/preds/no_llm_s1.csv`, `results/llemr/preds/llemr_ft_s1.csv` (test predictions, split 1) |
| `llemr_diagnostics/test_lactate_*.csv`, `test_value_shuffle.csv` | `results/llemr/diagnostics/` |
| `results/e2_s1.csv`, `results/e3_dose_s1.csv` | `results/e2/e2_s1.csv`, `results/e3/e3_dose_s1.csv` (rows of the Qwen variant `full` are ignored) |
| `numeric_sensitivity_excerpt.json` | `results/llemr/numeric_sensitivity.json` (the 9 probe texts and their cosines) |
| `recorded_numbers.json` | `report/BRIEF.md` sections 3 and 5, `results/llemr/llemr_ft_s1.json`, `llemr_ft_s1_ext.json`, `e2_s1.csv`, `llemr_preregistration.json` |

The patient record and the test predictions come from the PhysioNet/Computing in Cardiology Challenge 2012 data
(Silva et al., 2012; Goldberger et al., 2000), which is public on PhysioNet.

## Check the numbers

```bash
pip install scikit-learn      # optional: adds the cross-check of AUROC and AUPRC
python3 tools/verify_numbers.py
```

The script runs 92 checks: the reference values (typed from `BRIEF.md`) against the data files, against
scikit-learn, and against what the built lecture shows. It must end with `0 failure(s)`.

## Things to know before presenting

- **float32 vs. bf16.** The live logit of the patient is -4.9333; the project's own export (CPU, bf16) recorded
  -4.9307. The gap (0.003) is rounding noise. For the lactate-4 edit the live change is +0.0034 and the recorded one
  +0.0016: same sign, same tiny size.
- **The risks are uncalibrated** (the loss used a positive-class weight). Compare scenarios, not absolute probabilities.
- **The per-token shares in State 3 are an exact bookkeeping identity** (mean pooling plus a linear layer), not a causal
  attribution: attention mixes information between tokens before the mean.
- **One patient is an example.** The population statements (+1.08 and -0.12 percentage points) come from the recorded
  runs over 633 patients.
- **Scope.** One dataset, one split, one seed, one fine-tuning recipe for Llemr. The lecture does not claim that
  language models are worse in general; the continuation run is always called post hoc.
- Name and student ID are not in the lecture; add them to the cover page of the report.

## Files

```
01_icu_mortality.py     the lecture (only main() is traced)
_ehr.py, _plots.py      helpers; files starting with "_" are not traced by LecTrace
data/                   real data and recorded results (about 1.4 MB)
tools/export_weights.py checkpoint -> .npz without PyTorch
tools/verify_numbers.py independent check of every number
lectrace.toml, requirements.txt, .github/workflows/lectrace.yml   build and deployment
```
