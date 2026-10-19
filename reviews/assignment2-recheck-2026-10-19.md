# Assignment 2 follow-up review

Date: 19 October 2026. This reviews the updated files after the first audit. The original audit is preserved alongside this file as `assignment2-review-2026-10-19.md`; it records the earlier state, not the current state.

## Verdict

The principal report-structure and Table 1 fixes are present and verified. The three main Q1 physical claims were revised in the intended direction. Report dependencies are now documented, which is sufficient for a separate report-building environment, subject to testing that environment later.

The statement that everything from the full audit is fixed is still too broad. Some smaller findings were not in the chat summary and remain unchanged. The revised D screening addresses the epoch count, but not the original ambiguity about two seeds for every learned candidate. None of these observations requires retraining model A; phase 1 has not yet started.

## Confirmed fixes

1. `report/q2_part2.md` and `report/q2_part3.md` now exist. Across all three parts, every Q2.1-Q2.17 heading appears in order. The remaining answers are explicit placeholders rather than silently absent sections.
2. The sources now reference all six tables, all required figure groups and GIF-caption fields. Pending results and human answers are expected at this preparation stage.
3. Q2.5 and Q2.11 have `file:answers/...` inserts and captions with commit/start-time fields. The fields are populated from the phase logs when available. The answer files and logs themselves do not yet exist.
4. The group-equivariance argument is now in the final-model section. Its substitution g' = gT and g^-1 = T g'^-1 is correct.
5. Table 1 now escapes literal pipes. I passed a representative absolute-value row through the exact Pandoc format used by the builder and verified that the check, expected zero and actual equality result all survive in separate cells.
6. The README now documents Pandoc, Playwright and Chromium for report generation. Keeping those separate from training dependencies is reasonable when the PDF is built elsewhere. Verify that the final report environment has the required inputs, including phase logs and answers.
7. Q1 now limits information paths by graph direction, distinguishes a computational release path from physical arrival within the forecast hour, and ties release-discrepancy detection to arrival of its record.
8. D candidates now run for the trainer's default 30 epochs with seed 0 and validation-speed metrics. The selected seed-0 checkpoint and log use exactly the names expected for final model D seed 0, so the subsequent training loop skips that existing run and trains seed 1. Reuse is consistent with the stated final-epoch protocol.
9. Python parsing and macOS Bash syntax checks pass for the updated source tree. These are not end-to-end execution tests.

## Findings still open

### 1. Readiness reporting does not yet cover every gap [P2]

Source: `report/build.py`, particularly lines 32, 40, 101-127.

- On the actual empty checkout, `placeholders()` raises FileNotFoundError for `results/table1.json` before it can print NOT SUBMISSION READY. The same approach requires Figure 1 JSON keys to exist. The missing-results listing therefore does not work from the current preparation state.
- The final condition checks only named missing placeholders and absent report source parts. It does not check that image files exist, GIF references resolve, or the repository URL is real. In an isolated synthetic fixture, all placeholder fields could be filled with no missing markers while the URL still contained `/REPLACE/` and all nine referenced images were absent. Such a report could pass the current missing-marker condition.
- The warning is emitted after PDF creation. A missing rendering dependency can fail first, and an incomplete PDF can be emitted before the warning.

Suggested correction: gather missing required results, source parts, answers, image/GIF assets and metadata before rendering; reject obvious placeholder URLs; distinguish an explicit draft build from a submission-ready build. This is a robustness recommendation for preventing accidental incomplete submission, not an extra handout requirement about how the builder itself must be implemented.

I reproduced 32 missing markers with a minimal synthetic Table 1/Figure 1 fixture, so the field-warning mechanism does work once the unconditional inputs are supplied. Those fixture numbers are not experimental results and were never written to the project's results directory.

### 2. D screening is still single-seed [P2, interpretation risk]

Source: `run_phase3.sh`, lines 22-25, and the selection description in `report/q2_part3.md`.

The epoch problem is resolved. However, the first audit raised both 30 epochs and exactly two seeds for every learned model. Candidate selection still uses seed 0 only. The selected final D has both seeds, but the rejected learned candidates do not.

The handout explicitly permits validation-based architecture selection, so this remains an interpretation issue rather than evidence of test leakage. Calling the revised candidates "the full protocol" is stronger than the code supports under a literal reading of the two-seed rule. For the strictest approach, train each candidate for 30 epochs with seeds 0 and 1, select using a predeclared aggregate validation score, and reuse both runs of the winner. Alternatively clarify that single-seed validation screening is permitted and describe it accurately.

Compute accounting, without optional D+B:

| Workflow | Full 30-epoch runs | Total epochs |
|---|---:|---:|
| Current: A/B/C each twice, three D seed-0 candidates, winner seed 1 | 10 | 300 |
| Literal two-seed candidates: A/B/C each twice, three D candidates each twice, reuse winner | 12 | 360 |

The previous workflow used eight final runs plus three 10-epoch pilots, or 270 epochs. Thus the claim that the revised current workflow adds about one 30-epoch D run compared with that earlier workflow is arithmetically correct.

### 3. Smaller points from the full first audit remain unchanged

- `report/q1.md:104` still says approximately 1e8 multiply-adds take "a few milliseconds on one CPU core." There is no measurement supporting that latency. A qualified feasibility estimate is enough for the handout.
- `report/q1.md:34` still calls t0 the issuance time while allowing arrivals through t0+2 minutes and publication at t0+3 minutes. Label t0 as the nominal forecast origin and distinguish the snapshot/publication times explicitly. The rest of the design can retain the same cut-off rule.
- `pa2/evaluate.py:143` still decides scientific notation from the rounded magnitude. `fmt(0.0009999)` still returns `0.00100`. The literal handout rule for values below 1e-3 is best implemented by checking the original magnitude. This is a small presentation edge case.
- `report/title.md:8` still inserts a bare URL, not an explicit Markdown link. The previously reproduced plain-text rendering behavior therefore has not been changed at its source. A visible real URL may identify the repository, but explicit link markup is more reliable.
- `tests/` still has no `__init__.py`, while the README retains `python -m tests.test_windows`. The documented command failed to resolve in the older available environment in the first review. Verify it in the actual new `.venv`; add the package marker if needed to avoid a conflicting installed `tests` package.

### 4. Prediction insertion reads working-tree files [P2, final provenance check]

Source: `report/build.py:84-86`.

The inserts are now connected correctly, but they read current disk contents. The builder does not retrieve the Git blob associated with the logged commit or check that the answer still matches it. If the answer is edited after a phase starts, the report can describe changed text as unchanged while showing the earlier timestamp. The phase gates protect the files at experiment start; they are not a later report-build check.

Before final submission, either compare both answer files against their original prediction commits or insert those committed versions directly. Keep this check separate from whether the answers turned out to be correct.

## Expected unfinished work, not regressions

- No `.git`, `.venv`, trained checkpoint or experiment output exists yet.
- The actual repository URL is still the REPLACE value in `report/manual.json`.
- AI_PROMPTS and AI_EDITS remain to be filled. The disclosure is written in completed-work tense and should be finalized only after the described training, checks and commits actually happen. Include this review and any subsequent AI assistance as appropriate.
- The published flow measurements cannot be independently reproduced from this checkout until phase 1 generates its statistics. Do not use synthetic audit fixture values as results.
- Final report layout, wide tables, image readability, hyperlinks, captions and GIF paths remain to be checked in the rendered PDF.
- Q2.5 must be written and committed after phase 1 and before baseline stress tests; Q2.11 after phase 2 and before B-D training, including candidate runs. No answer to either question was generated or edited by this review.

## Practical next step

Phase 1 can proceed after setting up the intended environment and verifying the synthetic checks there. D-screening choices affect phase 3 and can be settled before that phase. The remaining report fixes should be completed before building the submission PDF. The original review and this follow-up are now both inside `reviews/`, so an agent restricted to this repository can read the full findings.

This follow-up used isolated synthetic fixtures and existing tools. It did not install dependencies, initialize Git, start training, run baseline stress tests, or change implementation files.
