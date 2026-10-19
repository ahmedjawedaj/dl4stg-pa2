# Assignment 2 implementation and submission review

Reviewed: 19 October 2026, Asia/Karachi.

Original specification: [13-page assignment handout](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/DL4STG_PA2 2/DL4STG_PA2.pdf>).

Implementation reviewed: `/Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2`.

## Verdict

The core forecasting implementation passed the synthetic checks performed in this review and mostly matches the specified experimental protocol on inspection. The supplied update overstates report completeness. The report sources stop at Q2.3, two required report parts are absent, and the builder can silently omit them. There are also reproducible Table 1 rendering and repository-link issues.

This is a preparation-stage project, not a completed submission. At review time there was no Git repository, `.venv`, trained checkpoint, generated result, figure/GIF from Q2, committed prediction answer, or final report PDF. These absences agree with the statement that training has not started. They do not by themselves indicate incorrect training code.

## Findings, in priority order

### F1. Required Q2 report sections are absent and silently skipped [P1]

Evidence: [report builder, lines 21-23 and 83-87](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/build.py:83>).

The builder lists `q2_part2.md` and `q2_part3.md`, but neither exists. Its `read()` returns an empty string for an absent source. Consequently, their omission produces neither a missing-placeholder warning nor a failure. Existing report sources contain only Q2.1, Q2.2 and Q2.3, and reference only Tables 1-2 and Figures 1-2. Q2.4-Q2.17, Tables 3-6, Figures 3-6, GIF captions, and the symmetry argument have no report sections yet.

The `file:` substitution mechanism exists, but no current report source invokes it for Q2.5 or Q2.11. Thus the claim that committed prediction answers are currently pasted into the report is not supported by the files on disk. Also, `file:` reads the working-tree file, not an explicitly recorded Git blob; the phase gates are the separate mechanism that checks for uncommitted changes.

Required correction: add the missing report parts, wire every required result and both prediction files into them, and fail before PDF creation if a required source, answer, table, figure, GIF reference, or field is absent. Preserve the original prediction-answer commits when completing the report.

Handout: pages 9-13.

### F2. Table 1 loses its values during Markdown conversion [P2]

Evidence: [Table 1 assembly, lines 34-36](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/build.py:34>).

Rows (a)-(c) contain literal absolute-value bars, such as `max|M(M(x)) - x|`. These bars are inserted into a pipe table without escaping. Using the builder's exact Pandoc input format, a representative row rendered as three cells containing `(a) max`, `M(M(x)) - x`, and an empty value. The expected zero and actual `torch.equal=True` result disappeared.

Required correction: escape literal pipes in table-cell text, or use math/HTML absolute-value notation that does not introduce table delimiters. Verify the rendered Table 1, not only the JSON values.

Handout: page 6.

### F3. Setup does not install the report-building dependencies [P2]

Evidence: [requirements](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/requirements.txt>) and [builder, lines 91-101](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/build.py:91>).

The builder requires the Python Playwright package, its Chromium browser, and Pandoc. The setup script installs the requirements file, which includes none of those report dependencies. Pandoc happens to be available on this Mac, but the inspected existing Python environments do not provide Playwright, and the intended `.venv` does not exist yet. A successful training setup therefore does not establish that the report can be built.

Required correction: document/install the report dependencies and check them before long training runs. Reproducibility would also benefit from recording the actual resolved package versions; the current requirements use lower bounds and unpinned packages.

Handout: page 1 requires one final PDF.

### F4. Q1 contains physical guarantees that its assumptions do not support [P2]

Evidence: [Q1, lines 70, 88, 102 and 104](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/q1.md:88>).

- A directed causal network does not let literally "any input in the window" affect "any output position". Accessibility depends on direction, hop count and time. For example, there is no S4-to-S1 path in the proposed graph.
- An eight-hour upper bound on reservoir-to-S4 travel time and a 24-hour history window do not guarantee that a release decided now physically affects the one-hour forecast target. Separate an available computational path from arrival of the physical release wave before the target time. The report's later S1 example, which explicitly compares travel time with the remaining 40 minutes, already makes this distinction correctly.
- Detection of an emergency release within one 10-minute cycle is not guaranteed when assumption A3 allows reporting delays of 20 minutes or outages of hours. State a delay-dependent detection time and what happens before detection.
- "A few milliseconds on one CPU core" is an unmeasured timing assertion. The handout requires a feasibility argument, not a benchmark. Use a qualified operation-count argument instead of a hard latency claim.

The six required Q1 headings and their principal sub-items are otherwise present. No explicit future measured rainfall or future measured water-level input appears in the proposal. Clarify that t0 is a nominal grid origin, the data cut-off is t0+2 minutes, and publication is t0+3 minutes, so these three times are not ambiguously described as the same issuance instant.

Handout: pages 2-3.

### F5. The D-selection runs introduce a protocol-compliance question [P2, interpretation risk]

Evidence: [phase 3, lines 22-25](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/run_phase3.sh:22>).

The final A-D runs use the required 30 epochs and seeds 0 and 1. However, the three D candidates are screened using 10 epochs and seed 0 only. The handout permits validation-based architecture selection, but also says every learned model uses the common 30-epoch protocol and exactly two seeds. It does not explicitly exempt exploratory candidates.

This review does not label validation-only screening as test leakage: the selection code uses validation results only. For the strictest interpretation, use the specified training protocol for candidate comparisons too. Otherwise disclose the screening procedure and obtain clarification on whether shorter, single-seed validation pilots are permitted. Do not present those pilots as the final reported D runs.

Current planned work is eight final runs plus three pilot runs: 240 final-run epochs + 30 pilot epochs = 270 epochs before optional D+B.

Handout: pages 8 and 11-12.

### F6. Submission metadata and AI disclosure remain unfinished [pending requirements]

Evidence: [manual fields](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/manual.json>) and [AI disclosure](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/ai_disclosure.md>).

The repository URL is still `https://github.com/REPLACE/dl4stg-pa2`. This is a populated string, so missing-placeholder detection will not catch it. AI_PROMPTS and AI_EDITS have no supplied values. The disclosure also describes training, evaluation, human checks and commits as completed events, although those events have not happened in this checkout.

Required completion: provide the actual repository URL, populate the prompts/outputs/edits disclosure, include this review and subsequent AI assistance where appropriate, and make the final disclosure accurately describe work actually performed. A private repository during development is fine, but it must be public at submission time. Do not claim any human checks until they are performed.

Handout: pages 1 and 13.

### Smaller reproducible issues

- The title page still emits a bare URL rather than an HTML anchor with the current Pandoc configuration. This contradicts the update's claimed hyperlink fix. A readable real URL may still communicate the repository location, but an explicit Markdown link is the reliable fix. See [title source](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/report/title.md:8>).
- [Number formatting](</Users/ahmedjawed/Documents/books/AI/MS/3rd Semester/Deep Learning for Time, Space and Graph/Assignments/assignment 2/dl4stg-pa2/pa2/evaluate.py:137>) checks the rounded value against 1e-3. Therefore `fmt(0.0009999)` produces `0.00100`, although the original value is below 1e-3. Compare the unrounded magnitude to implement the literal handout rule. Wrapped-model gaps are separately forced to scientific notation correctly.
- The README's `python -m tests.test_windows` did not resolve in the available Python environment, while executing the same test file through `runpy.run_path()` passed. Add `tests/__init__.py` or otherwise make the documented command unambiguous, then verify it in the new `.venv`.
- The specific flow numbers quoted in the supplied update cannot be independently checked from this checkout: no Figure 1 statistics or dataset are present. The script computes the relevant measurements, so compare the resulting JSON and figure with the prose after phase 1. Treat connected-component overlap labels as a heuristic, not ground truth.

## Q1 requirement map

| Required section, in handout order | Current coverage | Remaining action |
|---|---|---|
| 1. Architecture diagram | Present first, with input shapes, alignment/projection, graph and quantile outputs | Check final PDF legibility; the diagram has considerable white space and small labels when scaled |
| 2. Assumptions | A1-A11, with reasons and consequences; supplied facts separately identified | Keep location, travel-time and record-length assumptions clearly hypothetical |
| 3. Forecasting task/data | History, horizon, outputs, tensor table, axes, masks, timestamps and availability | Clarify nominal t0 versus actual cut-off/publication; keep the same rule everywhere |
| 4. Computations/justification | Projection, temporal/spatial equations, forecast head, loss and source paths | Correct the reach and physical-arrival claims in F4 |
| 5. Exceptional conditions | Missing, delayed, scheduled releases and schedule/actual discrepancy separately covered | Qualify emergency-detection and runtime claims |
| 6. Uncertainty/additional information | Unmonitored tributary, changing lags, extremes, rain, drift, releases, delays, and request table | Avoid presenting learned uncertainty widening or calibration as guaranteed |

Q1 requires a defensible written design; no river dataset, implementation or training run is required. Its proposed graph model is therefore not missing merely because it has no implementation.

## Q2 implementation map

| Handout requirement | Implementation evidence | Review status |
|---|---|---|
| Full Moving MNIST videos; split=None; whole-video indices | Data loader and split function | Matches on inspection; loader shape checked against current official torchvision source |
| float32 / 255 | Training and evaluation conversion paths | Matches |
| Normal-speed starts 1-11 | NORMAL_STARTS and sample_windows | Synthetic tests passed |
| Speed starts 1-14, 1-8, 1-2; uniform k then uniform t | AUG_STARTS and sample_windows | Synthetic tests passed |
| Five outputs, mask unavailable targets, equal contribution per video | Window functions and masked_video_mse | Synthetic mask/loss checks passed |
| Mirror, 90-degree anticlockwise rotation, eight symmetries/inverses | Shared transform functions | Exact checks and all inverse checks passed |
| Figure 1: first 50 train videos, 950 pairs, specified Farneback settings and mask | Figure 1 script | Matches on inspection; actual data results pending |
| Exact A architecture and default initialization | Model and cell definitions | 480,784 parameters independently verified |
| Nine steps, outputs 5-9, feedback gradients | Forecaster forward pass | Nine steps and nonzero feedback gradients independently verified |
| A-C have no prohibited extra layers | Same baseline architecture for A/B/C | Matches on inspection |
| Adam 0.001, batch 32, 30 epochs, final weights | Trainer defaults and final save | Matches for final phase-script runs; actual runs pending |
| Seeds 0 and 1 for every final learned model | Run scripts | Correct final-run loops; D pilot caveat in F5 |
| D has greater reach, <=721,176 parameters, no attention | Three model configurations | Counts and autograd reach independently verified |
| Validation-only D selection | Selection script reads validation training logs | Matches; no test-based candidate selection found |
| C/E metrics, speed E(2), valid-target handling | Evaluation module | Synthetic reference-predictor checks passed |
| Every learned result mean (seed 0, seed 1) | Table formatters and phase scripts | Implemented for errors/times; parameter count is one exact shared integer |
| Inference timing excludes device transfers, eval/no_grad; wrapper includes all eight passes | inference_time and Symmetrised | Matches on inspection; actual timings pending |
| Report hardware | Hardware script and builder field | Implemented; no hardware result generated yet |
| Optional D+B only after Table 5 justifies it | Separate --db path | Opt-in, not automatic; justification must be written after seeing Table 5 |
| Exact symmetry wrapper and comparison | Eight transform/predict/inverse/average loop | Synthetic probe consistency gaps below 3e-16; trained-model results pending |
| Committed Q2.5/Q2.11 before corresponding experiments | Phase 2/3 gates | Implemented; answer files and repository do not yet exist |

Official loader source: https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/moving_mnist.py . It transposes the raw array and adds the channel dimension, consistent with the loader's shape assertion; built-in splits select frames.

## Table and figure map

All result artifacts in this table are currently pending. The producer code exists; report integration is incomplete.

| Number | Required table and producer | Required figure and producer | Report integration |
|---|---|---|---|
| 1 | Transform/window checks: table1_checks | Flow direction/speed: fig1_flow | Source references exist; fix Table 1 escaping |
| 2 | A/Blank/Copy-last horizon errors: phase1_baseline | Three test videos, including overlap: phase1_baseline | Source references exist |
| 3 | A/Blank/Copy-last original and transformed errors/gaps: phase2_stress | Two worst M + two worst R videos, four rows, GIFs: phase2_stress + figs | Missing report section/captions |
| 4 | A/Copy-last E(2) all speeds and E(5) at 1x/2x: phase2_stress | Seed-0 C_M/C_R histograms with selected videos marked: phase2_stress | Missing report section/captions |
| 5 | A-D errors, gaps, parameters and times: phase3_final | One test video, all speeds, no overlap in displayed true frames: phase2_stress + figs | Missing report section/captions |
| 6 | Wrapped B/D and final model if different: phase3_final | Same Figure 3/5 videos with final model, plus GIFs: phase3_final + figs | Missing report section/captions |

For final PDF review, check the wide 12-column Tables 5-6 and stacked Figure 3/6 panels at actual page scale. Source code alone does not establish legibility or complete captions.

## Written-answer map

| Question | Requirement | Current report status |
|---|---|---|
| Q2.1 | Interpret direction/speed, 2x/3x and overlap bias | Draft exists; measurements pending |
| Q2.2 | Separate motion statistics and digit appearance | Draft exists; qualify finite-sample invariance claims |
| Q2.3 | Derive A's one-step reach | Draft exists; 8 px calculation checked |
| Q2.4 | Baseline versus references, horizon growth, overlap | Missing |
| Q2.5 | Pre-stress-test prediction, committed and reproduced unchanged | Missing; write after phase 1, before phase 2 |
| Q2.6 | Why low consistency gap alone is insufficient | Missing |
| Q2.7 | Precise motion failure on a selected Figure 3 video | Missing; requires inspecting actual predictions |
| Q2.8 | Rotation versus mirror and appearance explanation | Missing |
| Q2.9 | Speed breakdown and relation to reach | Missing |
| Q2.10 | Experiment separating unfamiliar speed from reach | Missing |
| Q2.11 | At most three pre-training sentences about B-D, committed unchanged | Missing; write after phase 2, before phase 3 |
| Q2.12 | Describe chosen D and demonstrate greater reach | Missing; checked candidate calculations are available |
| Q2.13 | Attribute failures using comparisons in Table 5 | Missing |
| Q2.14 | Original 1x accuracy cost/trade-off | Missing |
| Q2.15 | Learned versus built-in symmetry, accuracy, cost and blur | Missing; also include the required short equivariance argument |
| Q2.16 | Revisit Q2.7 video using final-model Figure 6 | Missing |
| Q2.17 | Motion verdict, skeptical evidence and one unpassed test; <=150 words | Missing |

## Independent checks performed

- Parsed all 17 Python files and checked all four shell scripts with both macOS `/bin/bash` and Homebrew Bash: passed syntax checks. This does not establish full execution on either shell.
- Ran the existing window/symmetry test file in an isolated temporary directory: passed.
- Ran the handout's Table 1 checks using the implementation's functions: exact identities, marker coordinates and frame selections matched.
- Ran the existing autograd reach checker for all four configurations:

| Architecture | Trainable parameters | One-step reach |
|---|---:|---:|
| A | 480,784 | 8 px |
| D_k5 | 709,520 | 16 px |
| D_dil | 703,888 | 12 px |
| D_p8 | 695,584 | 16 px |

- Checked nine-step output shape and nonzero gradient to each fed-back predicted frame: passed.
- Checked two videos with different valid-target counts receive equal loss weight: expected and actual loss 2.5.
- Checked Blank and Copy-last evaluation against independent synthetic per-horizon calculations at all speeds: passed, including exactly zero mirror/rotation consistency gaps.
- Checked symmetrisation on a synthetic asymmetric predictor: mirror gap 2.18e-16, rotation gap 2.35e-16.
- Reproduced Table 1 cell loss and plain-text title URL using the actual Pandoc input format.
- Confirmed no report source currently includes either prediction-answer file.

Runtime limitation: synthetic tensor checks used an existing Python 3.9.13 / PyTorch 1.9.0 / torchvision 0.10.0 / NumPy 1.24.4 / OpenCV 4.6.0 installation. That installation predates the project's requirements and cannot validate the current MovingMNIST API or the intended MPS training environment. The new `.venv`, end-to-end dataset loading/training, measured flow results, timings and final PDF remain unverified. Tests were isolated and did not populate the project's results directories, download data, train models, or run learned-model stress tests.

## Completion order

1. Repair report completeness/escaping, add report dependencies, and settle the D-screening protocol. Preserve genuine prediction ordering.
2. Set up the intended environment and rerun synthetic checks there. Keep PA2_SMOKE unset for actual reported runs.
3. Initialize Git and create the actual remote. Run phase 1: Table 1, Figure 1, reach, both full A runs, Table 2 and Figure 2.
4. Inspect phase 1 outputs and write Q2.4. Write and commit the genuine original Q2.5 before any baseline stress experiment.
5. Run phase 2. Inspect Figures 3-5 and write Q2.6-Q2.10.
6. Write and commit Q2.11, at most three sentences, before training B-D including design pilots.
7. Run phase 3. Complete Tables 5-6 and Figure 6; choose optional D+B only if Table 5 supports it and explain the choice. Include the equivariance proof and Q2.12-Q2.17.
8. Fill the real repository URL and AI disclosure, verify verbatim prediction-answer inclusion, and build one PDF. Check every required question, table, figure, caption and GIF path in the rendered output.
9. Make the repository public at submission time. Submit the PDF through the LMS by Monday, 26 October 2026, 11:55 pm. The handout specifies a 25% one-day late penalty, 50% for the second day, and no acceptance afterward.

This review did not edit the implementation or start training.
