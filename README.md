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

Setup: `./setup_env.sh` creates `.venv` and installs `requirements.txt`. Tests: `.venv/bin/python -m tests.test_windows`.

Report: `python report/build.py` builds `report/DL4STG_PA2_report.pdf`. It needs `pandoc` and
`pip install playwright && playwright install chromium` (not in `requirements.txt`, which covers the
experiments only). It lists every gap (missing part, answer, table, image, GIF, repository URL, or a prediction answer edited after its commit) before rendering, and `python report/build.py --final` refuses to write a PDF while any gap remains.
