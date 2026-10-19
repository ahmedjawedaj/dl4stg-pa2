# DL4STG PA2: Motion or Memory? (ConvLSTM on Moving MNIST)

AI651 Deep Learning for Space, Time and Graphs, Assignment 2. Code for Question 2 and the report.

* `pa2/`: data split and windows (`data.py`), ConvLSTM models (`models.py`), training protocol
  (`train.py`), metrics (`evaluate.py`), figure helpers (`plots.py`, `figs.py`).
* `scripts/`: one script per table or figure group.
* `run_phase1.sh`, `run_phase2.sh`, `run_phase3.sh` run the experiments in the order the handout
  requires. Phase 2 refuses to run until `answers/Q2_5.md` is committed, and phase 3 until
  `answers/Q2_11.md` is committed. `./run_phase3.sh --db` adds the optional D+B model.
* `checkpoints/` final-epoch weights, `results/` tables and metrics, `figures/` and `gifs/` outputs,
  `logs/` run logs, `report/` report sources and the PDF.

Run all commands from the repository root. Use Python 3.10 or newer and Git. The completed
experiments used Python 3.13.7, PyTorch 2.14.1 and an Apple M1 Max with 64 GB RAM on MPS.
CUDA and CPU are selected automatically when appropriate. CPU training will be slower.

```bash
./setup_env.sh
.venv/bin/python -m tests.test_windows
.venv/bin/python -m tests.test_resume
.venv/bin/python -m scripts.download_data
```

The dataset download is about 819 MB. The supplied results and PDF can be read without
installing the environment or downloading the data.

The setup uses official PyPI without changing machine-wide pip configuration and records exact
versions in `logs/pip_freeze.txt`. `requirements.txt` lists the experiment dependencies;
the freeze file records the exact versions used for the submitted runs.
Training writes an atomic resume checkpoint and progress JSON after every complete epoch.
Rerunning the same phase resumes an interrupted run with its optimizer and sampling state intact.
Final checkpoint filenames are written only after all requested epochs finish.

The download command fetches the official Moving MNIST file over HTTPS
with four byte-range transfers, reuses completed chunks, and verifies its published size and MD5.
Experiments still load the verified file through torchvision with `split=None`.

To reproduce evaluation and figures using the supplied final checkpoints, run these in order
in a separate clone. They overwrite generated results and timings in that clone.

```bash
.venv/bin/python -m scripts.phase1_baseline
.venv/bin/python -m scripts.phase2_stress
.venv/bin/python -m scripts.phase3_final
```

To repeat training, move the supplied `checkpoints/` directory aside in that separate clone
first. Then run `./run_phase1.sh`, `./run_phase2.sh` and `./run_phase3.sh` in order. Phase scripts
skip existing final checkpoints and resume incomplete runs. The selected D weights have names
such as `D_D_dil_s0.pt`: the first D is the experiment model and `D_dil` is the architecture.

`./run_all.sh` coordinates those phases, but pauses for committed prediction answers and a
Table 5 decision in `results/db_decision.json` (`train_optional_DB` must be a JSON boolean).
It does not write those judgments itself. This submission's decision is false; a new experiment
needs a fresh decision based on its own Table 5. The supplied prediction commits and logs
document the original experiment chronology. A new assignment attempt must make its own
predictions before the corresponding experiments, without previewing their results.

To rebuild the report, install Pandoc separately and the report-only Python dependencies:

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python report/build.py
```

This builds `report/DL4STG_PA2_report.pdf` and lists missing content or assets, including any
prediction answer edited after its commit. `.venv/bin/python report/build.py --final` refuses
to render while any gap remains. Editable Q2 result answers and the AI-prompt section are in
`report/manual.json`; Q1 and the other report text are in `report/*.md`. Preserve `answers/Q2_5.md`
and `answers/Q2_11.md` exactly as originally committed. Inspect the rendered PDF before submission.

To use an already-installed Chromium instead of downloading another browser, set
`PA2_CHROMIUM_EXECUTABLE` to its executable path when invoking the builder.
If Pandoc is installed outside the current PATH, set `PA2_PANDOC` to its executable path.

See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for the mapping from the original handout
to every answer, table, figure and submission requirement. All 12 planned 30-epoch runs are
complete. Validation selected `D_dil`, and both selected candidate checkpoints are reused.
The final model is symmetrised D. Optional D+B was not added, with the Table 5 decision recorded
in `results/db_decision.json`.

The report retains both seeds and the weak D seed-0 illustrations. Exact symmetry fixes
consistency but does not establish accurate motion. The original Claude-prompt section is a
clearly marked draft placeholder that Ahmed will complete before submission. All measurements,
the Q2.5/Q2.11 prediction commits and the sleep/wake timing record are preserved.
