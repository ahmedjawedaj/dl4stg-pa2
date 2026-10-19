# Assignment 2 execution log

## Authorization and constraints

Ahmed asked Codex: "please take it over and do whats needed" after reviewing the implementation and the proposed phased workflow.

- Finish the assignment locally and preserve genuine experiment ordering.
- Keep the existing local Git history. Do not create or publish a GitHub repository until the work is ready.
- Q2.5 must be written and committed before any learned-baseline mirror, rotation or speed test.
- Q2.11 must be written and committed before B-D training, including validation candidates.
- All final learned runs use 30 epochs, batch 32, Adam 0.001, seeds 0 and 1, final weights, and the original rollout/loss.
- D selection compares all three candidates using both seeds and validation data only, reusing both winner checkpoints.
- Do not fabricate numerical results, original Claude prompts, or claims that Ahmed personally checked outputs.

## Current takeover work

1. Verified local Git exists with initial commit `1b1436c`. No remote is configured.
2. Found a real installation problem: global pip configuration includes an obsolete NVIDIA index that fails DNS resolution. The setup now ignores global pip configuration for its own installation commands and uses official PyPI. Global configuration is unchanged.
3. Created the intended Python 3.13 virtual environment. PyTorch 2.14.1, torchvision 0.29.1, NumPy 2.5.4 and OpenCV 5.0.0 installed successfully; MPS is available on the Apple M1 Max with 64 GB RAM.
4. Downloaded the official Moving MNIST file using HTTPS byte ranges, retaining the original partial transfer. The assembled file passed the exact 819,200,096-byte size and MD5 `be083ec986bfe91a449d63653c411eb2` checks; provenance is in `results/dataset.json`.
5. Added per-epoch progress JSON and a separate atomic resume checkpoint. Resume restores optimizer, window/augmentation RNG and CPU RNG, and rejects changes to seed, model, data count or runtime. Only a completed run produces the final checkpoint filename that phase scripts skip.
6. Added an isolated test that compares uninterrupted training against an interruption/resume, checks identical final weights/losses, and rejects a seed mismatch. It passed in the new environment, as did the existing window/symmetry tests. A complete synthetic one-batch MPS trainer run, including final checkpoint serialization, passed. Synthetic batches are not assignment results.
7. Table 1, hardware recording and autograd reach checks ran successfully in the intended environment. The latter confirmed A=8 px / 480,784 parameters and all three larger-reach D configurations within the limit.
8. Installed report-generation dependencies and tested a draft PDF using the existing bundled Chromium. The builder accepts `PA2_CHROMIUM_EXECUTABLE` to avoid another browser download. Missing results remain visibly marked; the draft is not submission ready.
9. Reworked the Q1 architecture graphic after visual review showed the old diagram's labels were too small at printed size. The updated single-column graphic fits one page and was rendered and inspected.
10. Committed setup, reliability changes and initial verification artifacts as `64166c1`.
11. Started detached phase 1 at 2026-10-19 16:55:50 Asia/Karachi (11:55:50 UTC), PID 16049. Read `logs/active_run.json` and `logs/run_phase1.log` before any new launch. `caffeinate -i` prevents idle sleep during this job. Wait for both A seeds and `PHASE 1 DONE` before Q2.5; no learned-model stress tests have been run yet.
12. The first real A seed-0 epoch completed on all 8,000 videos: training loss 0.0461222, validation E(5) 0.0393209, training time 16.7021 seconds. The process continued to epoch 2. This is a first-epoch training-time observation, not a complete runtime benchmark; phase 1 is roughly a 20-minute job if this rate holds, and the remaining work is provisionally a few hours, subject to the larger D models and evaluation costs.
13. Figure 1 was generated and visually inspected from 950 training pairs and 103,050 retained pixels. Median flow speed is 3.31750 px/frame, giving typical 2x/3x displacements of 6.63500/9.95251 px. Overlap heuristic fraction is 0.609093, with medians 3.30176 (overlap) and 3.50651 (separate). Full measured statistics are in `results/fig1_flow_stats.json` and match the earlier qualitative draft; use the measured JSON, not audit fixtures, for writing.

## Next actions

1. Continue the existing phase 1 process. Do not launch another training job. Environment, dataset, synthetic checks, MPS trainer check, Table 1, Figure 1 and reach verification are complete.
2. After both A seeds and phase 1 evaluation finish, inspect Table 2/Figure 2, write Q2.4, then write/commit a genuine Q2.5 before phase 2.
3. After phase 2, inspect actual failures and complete Q2.6-Q2.10. Write/commit Q2.11 before B-D training.
4. Run phase 3, make the optional D+B decision from evidence, complete the remaining writing, and render/verify the PDF.
5. Ahmed answered that a dummy may be added for now. A clearly labelled temporary placeholder is in AI_PROMPTS, with an explicit missing marker so final submission readiness cannot mistake it for the actual Claude transcript. The known Codex takeover request and continuation authorization are quoted accurately. Do not repeatedly ask for the transcript during training; replace the temporary section before submission.
6. Create/publish the GitHub repository only when the local work is ready, then insert its actual URL and perform final submission checks.

Ahmed explicitly answered "Yes, continue automatically". The thread follow-up `finish-assignment-2` is ACTIVE every 10 minutes and should notify only on meaningful milestones, failures, completion or required user action. It must be disabled when the authorized assignment work is complete. Keep the Mac on and this desktop app running for local continuation.

The working report build command on this Mac is the project Python running `report/build.py` with `PA2_CHROMIUM_EXECUTABLE` pointing to `/Users/ahmedjawed/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`.

## Latest automatic check

2026-10-19 17:15:07 PKT: phase 1 is healthy. A seed 0 finished all 30 epochs and its final checkpoint loaded successfully with 480,784 finite parameters. Its final validation E(5) is 0.0221035 and mean training epoch time is 26.55 seconds. The per-epoch resume file was removed after finalization. A seed 1 has completed 9/30 epochs, latest validation E(5) 0.0300264. Expect roughly another 9-12 minutes for seed 1 and phase 1 evaluation, subject to actual runtime. No test-based analysis, stress experiments or prediction commits have happened yet. Keep waiting for both final checkpoints and PHASE 1 DONE before Q2.5.

## Phase 1 complete and Q2.5 prepared

Both A seeds completed 30 epochs. Mean test E(5) = 0.0219346, versus Blank 0.0425645 and Copy-last 0.0640821. Figure 2 was visually inspected on videos 9007, 9012 and 9000 (overlap). Q2.4 was written from those results. The genuine Q2.5 prediction was written before phase 2, using only phase 1 evidence, and its permitted inputs were hashed in results/q2_5_preexperiment_evidence.json. Commit this answer before launching phase 2; preserve it unchanged regardless of the result.

## Phase 2 started

Q2.5 was committed as `2513a26` before any stress test. Phase 2 launched at 2026-10-19T17:28:23.156129+05:00 with PID 39412. Read logs/run_phase2.log and logs/phase2_gate.txt. No B/C/D training is authorized by the gate until a genuine Q2.11 has been written and committed after phase 2.

## Phase 2 complete and Q2.11 prepared

Phase 2 finished normally after the Q2.5 gate. Mean A C_M=0.0017784 and C_R=0.0028035, transformed E_M=0.0225576 and E_R=0.0236072. Mean E(2) across 1x/2x/3x is 0.0185294/0.0454917/0.0533584. Figures 3-5 were visually inspected. Q2.11 contains three genuine prediction sentences using only phase 2 evidence, hashed in results/q2_11_preexperiment_evidence.json. Commit it before phase 3 and preserve it unchanged. Figure 3 GIF metadata exposed a zero-duration compatibility bug; fix playback timing without changing the measurements.
