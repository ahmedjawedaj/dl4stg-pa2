# Assignment 2 execution log

## Current status

Ahmed authorized a performance-improvement attempt on 20 October 2026. The original twelve runs and their report are complete and preserved in Git at 995521b. Add exactly one D candidate, D_dil64, with C's 64-channel width and only layer-2 dilation changed to 2. Train both seeds for 30 full epochs under the unchanged protocol. Read experiments/d_refinement_plan.md. Select across all four candidates using the existing validation-only score before any new test evaluation. The original test results were already inspected; disclose the later adaptive experiment honestly. Do not change Q2.5/Q2.11 or discard earlier runs. Inspect logs/active_run.json and processes before any launch. After selection, update affected evaluation/report artifacts if the winner changes, otherwise record the unsuccessful attempt. The published PDF currently describes the original three-candidate experiment and will need the extension documented. Original Claude prompts remain explicitly deferred by Ahmed; do not ask again.

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

## Phase 3 started

Q2.11 was committed as `fefd535` before any B/C/D training. Phase 3 launched at 2026-10-19T17:29:25.744548+05:00 with PID 40259. It will train B and C on both seeds and all three D candidates on both seeds, 30 epochs each, then select D by validation data and evaluate the final wrappers. Read logs/run_phase3.log and logs/active_run.json before any launch. No optional D+B run is selected yet.

## Phase 2 report and presentation check

Q2.6-Q2.10 are filled from saved metrics and visually inspected cases. Q2.7 selects mirror video 9658 for the final Figure 6 comparison. Fixed a real GIF compatibility error: ImageIO duration is milliseconds in this environment, so 0.9 produced zero-delay frames. All four baseline GIFs now have five unchanged decoded frames with 900 ms delay, and the generator uses 900 for final GIFs. Figure 4 tick labels are now sparse and readable. Report panels now paginate individually, with a global image-width constraint to prevent print clipping and unwanted global print scaling. Tables 5/6 have landscape pages reserved. Rebuilt the draft and visually inspected pages containing Q1 diagrams and phase 2 figures/tables; remaining draft gaps are expected phase 3 outputs, disclosure, and repository URL.

At 2026-10-19T17:34:10.141695+05:00, B seed 0 has 11/30 epochs, mean training epoch 22.48 s. The ten B/C/D runs provisionally need roughly 2-3 hours including validation/evaluation, subject to actual D timings, followed by report completion. Continue the existing PID in logs/active_run.json. No GPU evaluation was run alongside training.

## Q1 compliance audit during phase 3

Read the complete Q1 brief and matched its six required headings to the proposal. Removed unsupported geographic claims inferred from the schematic, labelled the 8 h / 12 h routing bounds as provisional, and made extension of the window conditional on historical timing. Clarified that level and schedule have explicit ages while rainfall and actual release use the stated mask histories, keeping the 14-channel tensor specification consistent. Replaced an unmeasured subsecond claim with a conservative arithmetic estimate and deployment latency check. Added the edits actually performed to AI_EDITS, leaving the original Claude-prompt placeholder explicit. Rebuilt the draft, confirmed the missing-result checks still block final submission, and visually inspected the updated feasibility page.

2026-10-19T17:37:57.268422+05:00: existing phase 3 process remains healthy, B seed 0 19/30 epochs, mean training epoch 25.23 seconds. No duplicate training or GPU evaluation launched. Remaining work: finish B/C and six D candidate runs, inspect final metrics and choose optional D+B only from Table 5, complete Q2.12-Q2.17, final disclosure/repository link and full rendered-PDF review.

2026-10-19T17:46:12.360581+05:00: B seed 0 completed all 30 full-data epochs, final checkpoint loaded on CPU and no resume file remains. Mean training epoch 27.09 s. The existing phase 3 script automatically advanced to C seed 0, which has completed four epochs. No test-based B/C conclusions have been drawn yet. Process and logs are healthy.

2026-10-19T17:55:56.648740+05:00: phase 3 healthy, C seed 0 has 24 completed epochs and is finishing epoch 25. Existing PID 40259 remains active. Mean C training epoch 28.56 s. No intervention needed and no duplicate jobs launched.

2026-10-19T18:05:59.547215+05:00: C seed 0 completed 30 full-data epochs; final checkpoint loads on CPU with finite tensors and resume file is removed. Mean training epoch 28.57 s. Existing phase 3 automatically advanced to B seed 1 (15/30 complete at inspection). No test evaluation launched alongside training.

2026-10-19T18:15:59.580689+05:00: both B seeds are complete. B seed 1 final checkpoint verified on CPU with finite tensors after 30 full-data epochs, mean training epoch 28.44 s. Existing phase 3 advanced to C seed 1, five epochs complete at inspection. All six D candidate runs remain queued. No intervention needed.

2026-10-19T18:25:51.975588+05:00: phase 3 remains healthy, C seed 1 has 21 completed epochs and is in epoch 22. Its measured mean training epoch is 33.99 s. Existing PID 40259 is continuing normally; D candidates will start automatically after this run. No intervention or duplicate launch.

2026-10-19T18:36:10.845343+05:00: both B and C are complete on both seeds. C seed 1 final checkpoint verified on CPU with finite tensors after 30 full-data epochs, mean training epoch 33.84 s. Existing phase 3 advanced to D_k5 seed 0 (3/30 complete), averaging 98.34 training seconds/epoch. This is substantially slower than A-C. The two D_k5 runs alone require about 100 minutes plus validation, and the remaining D_dil/D_p8 timings are not yet measured. Provisionally allow another 2-3 hours for D screening and final evaluation, then report completion; refine from actual candidate timings. No protocol changes or duplicate jobs.

2026-10-19T18:45:57.883992+05:00: D_k5 seed 0 has 10 completed epochs and is in epoch 11. Mean training epoch is now 84.79 s (recent epochs 73-77 s). The job is running normally with finite logged losses, although validation E(5) has plateaued near 0.03947. Preserve the full fixed 30-epoch protocol and all candidates; do not tune or select from test results. No intervention needed.

2026-10-19T18:55:58.590608+05:00: D_k5 seed 0 is healthy at 17 completed epochs, currently epoch 18. Mean training epoch 83.74 s. Existing phase 3 PID 40259 continues without errors. Fixed protocol and candidate selection remain unchanged; no new launch.

2026-10-19T19:06:03.101505+05:00: D_k5 seed 0 has 23 complete epochs, mean training epoch 87.67 s (recent about 100 s). Existing phase 3 process and resume progress are healthy. Continue the fixed run and validation-only selection without intervention.

2026-10-19T19:15:55.591103+05:00: D_k5 seed 0 has 28 complete epochs and is in epoch 29, mean training epoch 90.46 s. Existing process and per-epoch recovery are healthy. Expect automatic seed-1 transition after the full run and validation scoring. No intervention.

2026-10-19T19:26:02.799411+05:00: D_k5 seed 0 completed all 30 full-data epochs, final checkpoint loaded on CPU with finite tensors, resume removed. Mean training epoch 91.83 s. Existing phase 3 advanced to D_k5 seed 1, 3 complete epochs at inspection (mean 110.36 s). Retain the observed validation plateau as an actual candidate outcome; do not discard or rerun with changed settings. The remaining two designs will still be compared on both seeds and validation only.

2026-10-19T19:27:59.110423+05:00: Ahmed said he is about to travel home and may put the MacBook to sleep. At inspection D_k5 seed 1 had four completed epochs and was in epoch 5, with an atomic resume file present. Sleep is planned, not yet observed or confirmed. After wake inspect the actual process first. If the MPS job failed, resume the same phase from the saved epoch without changing protocol or launching a duplicate. Record any observed interruption honestly and check its effect on timing before reporting benchmark means.

## Observed sleep and successful continuation

2026-10-19T21:16:05.852373+05:00: the queued heartbeat ran after wake. macOS power logs confirm lid/deep sleep with periodic dark wakes and a full lid/HID wake at 2026-10-19 21:14:51 +0500. Saved relevant power events to logs/sleep_wake_2026-10-19.txt. The original phase 3 parent (40259) and D_k5 seed-1 process (10075) survived and continued through 10 epochs, now in epoch 11. No restart, lost completed epochs, or duplicate process. Results/sleep_interruption.json records this observation. Preserve raw epoch timings and disclose this run's sleep/wake interruption wherever relevant; do not fabricate an uninterrupted benchmark or subtract guessed sleep overhead. Continue existing training.

2026-10-19T21:18:07.167678+05:00: post-wake training continues on the original process. D_k5 seed 1 has 14 completed epochs and is in epoch 15, recent training epochs about 34 s and validation E(5) about 0.0344. No recovery or intervention needed. Sleep/wake timing disclosure remains required for this candidate.

2026-10-19T21:28:33.584281+05:00: D_k5 seed 1 has 24 complete epochs and is in epoch 25. Validation E(5) has improved to 0.02973; preserve this measured outcome alongside the seed-0 plateau. Original phase 3 process remains healthy and will advance automatically. No intervention or new GPU job.

2026-10-19T21:38:38.916319+05:00: D_k5 completed both 30-epoch seed runs. Seed-1 final checkpoint loads on CPU with finite tensors and the resume file is gone. Raw mean training epoch 72.58 s, with the recorded sleep/wake interruption. Seed-1 validation E(2) is {'E2_1x': 0.02595994520187378, 'E2_2x': 0.02905267024040222, 'E2_3x': 0.031652369976043704, 'E5_1x': 0.02925631022453308}. Existing phase 3 advanced to D_dil seed 0, two epochs complete at about 63.24 s per training epoch. Keep both remaining designs and both seeds; no candidate selected yet.

2026-10-19T21:48:30.515004+05:00: D_dil seed 0 has 10 completed epochs and is in epoch 11, mean training epoch 69.45 s. Existing process is healthy with finite logged losses. Continue the queued fixed-protocol runs. No intervention or duplicate job.

2026-10-19T21:58:29.997982+05:00: D_dil seed 0 has 17 complete epochs, mean training epoch 75.05 s; latest validation E(5) 0.03819. Existing process is healthy and advancing. Continue fixed protocol with no duplicate job or test-based selection.

2026-10-19T22:08:44.433576+05:00: D_dil seed 0 has 24 complete epochs and is finishing epoch 25, mean training epoch 74.85 s. Existing process is healthy. A read-only publication preflight confirmed GitHub CLI is installed and authenticated as ahmedjawedaj with repository access. No repository was created, remote added or content uploaded; publication still waits until local work is ready.

2026-10-19T22:18:38.237518+05:00: D_dil seed 0 completed all 30 full-data epochs, final checkpoint verified on CPU with finite tensors and resume removed. Mean training epoch 73.12 s. Final validation: {'E2_1x': 0.03794672584533691, 'E2_2x': 0.03791957712173462, 'E2_3x': 0.03791330099105835, 'E5_1x': 0.037922581434249876}. Existing phase 3 advanced to D_dil seed 1, three epochs complete at about 59.68 s per training epoch. Both D_p8 runs remain queued; no model selection or extra GPU job.

2026-10-19T22:28:32.012983+05:00: D_dil seed 1 has 12 complete epochs and is in epoch 13. Mean training epoch 66.05 s, latest validation E(5) 0.03491. Existing phase 3 is healthy; continue the queued runs without intervention.

2026-10-19T22:38:30.988035+05:00: D_dil seed 1 has 20 complete epochs and is in epoch 21. Mean training epoch 67.45 s, latest validation E(5) 0.02884. Existing process and resume progress remain healthy. No intervention or duplicate run.

2026-10-19T22:49:43.755031+05:00: D_dil completed both 30-epoch seeds. Seed-1 final checkpoint verified on CPU with finite tensors and resume removed. Mean training epoch 66.71 s. Final validation {'E2_1x': 0.023623838663101197, 'E2_2x': 0.02846138286590576, 'E2_3x': 0.03177223324775696, 'E5_1x': 0.026870853185653687}. Existing phase 3 advanced to D_p8 seed 0, the last candidate design. Added the observed sleep/wake timing context to the report. A changed desktop app environment no longer includes Pandoc on PATH, although /opt/anaconda3/bin/pandoc is present, so the builder now accepts PA2_PANDOC for an explicit executable path. Use this alongside PA2_CHROMIUM_EXECUTABLE in future builds.

2026-10-19T22:59:26.159245+05:00: D_p8 seed 0 completed all 30 full-data epochs, final checkpoint verified on CPU with finite tensors and resume removed. Mean training epoch 12.24 s. Final validation {'E2_1x': 0.03794570255279541, 'E2_2x': 0.03791915369033814, 'E2_3x': 0.037913577318191526, 'E5_1x': 0.03792135763168335}. Existing phase 3 advanced to the last required training run, D_p8 seed 1, 15/30 complete at inspection. Selection and final evaluation remain automatic after completion.

## All required training complete

2026-10-19T23:09:16.204097+05:00: D_p8 seed 1 completed all 30 full-data epochs, checkpoint verified on CPU with finite tensors and resume removed. All twelve required runs are complete. Validation-only averaging selected D_dil, reusing both candidate runs. The existing phase 3 script is evaluating original and symmetrised models. Q2.12 was written from the selected architecture and recorded reach check. No optional D+B decision until Table 5 is complete. Initial D seed-0 test metrics show weak, near-constant errors across horizons; preserve this outcome and inspect Figure 6 before interpreting it.

2026-10-19T23:16:01.028642+05:00: all required training remains complete. The existing phase3_final process is evaluating the last wrapped D seed; symmetrised D seed 0 has finished with near-rounding-level gaps and E(5)=0.038539. Symmetrised D evaluation is slower than the learned models because of the eight forward passes, so leave the existing job running. Q2.12-Q2.14 are filled from selected architecture and actual A-D measurements. Added SUBMISSION_CHECKLIST.md mapping every handout item and submission gate to repository evidence. Corrected the inference-timing description to match the actual three-batch warm-up (the handout specifies no full-pass warm-up requirement). An async question asks Ahmed to provide the actual Claude prompts or confirm that they are unavailable, so the final disclosure can be truthful. Do not invent them. GitHub name ahmedjawedaj/dl4stg-pa2 is currently available according to a read-only lookup; no creation/upload yet.

## Phase 3 and scientific writing complete

2026-10-19T23:18:22.445853+05:00: phase 3 finished normally, Tables 5/6 and Figure 6/GIFs exist, and there is no training process left. Inspected the final mirror, rotation and speed panels: seed 0 produces a weak near-uniform field, which is preserved and described honestly. Wrote Q2.15-Q2.17 and the no-D+B decision from the completed Table 5, with its hash in results/db_decision.json. Q2.17 is 122 words. Ahmed replied that he will do the original-Claude-prompt disclosure later. Leave its explicit draft placeholder, do not ask again, and finish all remaining automatic work, PDF checks and end-stage repository preparation.

2026-10-19T23:23:49.889792+05:00: after completing local results/writing and visual draft checks, created the public GitHub repository https://github.com/ahmedjawedaj/dl4stg-pa2. No content uploaded yet. Inserted its actual URL into the report and rebuilding the reviewed draft. Original Claude prompts remain deferred to Ahmed.

## Reviewed deliverable ready for upload

2026-10-19T23:27:18.003141+05:00: all 26 pages were rendered and visually inspected, with full-size checks of the equations, wide Tables 5/6, prediction panels and AI disclosure. Fixed the isolated assumption row, excess disclosure spillover and an equation heading separated from its equations. Verified the actual clickable public-repository URL on page 1, no image overflow, all 17 Q2 answers, eight five-frame GIFs at 900 ms per horizon, the twelve complete runs and genuine prediction chronology. Final strict build correctly refuses only the original-Claude-prompt field that Ahmed explicitly deferred. The reviewed draft is copied to the workspace output/pdf/DL4STG_PA2_report_draft.pdf. Source and PDF SHA256 are recorded in results/submission_audit.json. Publishing the completed local repository next; no LMS submission is made.

2026-10-19T23:28:12.641449+05:00: public upload succeeded. Verified repository visibility PUBLIC, main matches the local commit, and the uploaded PDF Git blob exactly matches the reviewed local PDF. All automatic work is complete. Only Ahmed's explicitly deferred original-Claude-prompt disclosure remains before submission. The strict final guard continues to flag that field. Stopping the now-unneeded recurring training checks next.

2026-10-19T23:29:09.026728+05:00: finish-assignment-2 follow-up is confirmed PAUSED after all automatic work completed. The only deferred task belongs to Ahmed: replace the original-Claude-prompt placeholder before submission. No LMS submission was made. The reviewed draft and public repository are ready for his review.

## Final readiness review requested by Ahmed

2026-10-19T23:52:29.716346+05:00: rechecked the original handout, report, training/evaluation implementation and README. Found run_all.sh still waited for an obsolete optional-D+B text file, which would hang a reproduction after phase 3. Fixed it to read the actual results/db_decision.json boolean, with invalid-value rejection. Isolated wrapper cases (false/true/invalid) and shell syntax passed, along with existing window and resume checks. Reverified all twelve 30-epoch full-data logs, 1,000-video metric summaries, D parameter/reach limits, validation-only winner and decision hash. Expanded README with executable venv commands and reproduction/report guidance. No experiments rerun or report/PDF sources changed. reviews/final_readiness_review.md records the candid verdict: 26 pages are permissible but not required, original Claude prompts remain deferred, and weak D seed 0 limits the scientific claim.

## Authorized D refinement

2026-10-20T13:18:11.357693+05:00: Ahmed requested performance improvement before submission. Registered D_dil64 and the fixed two-seed, 30-epoch validation-only comparison in experiments/d_refinement_plan.md before training. Original selection copied to results/d_selection_initial.json/.md and all original artifacts retained in Git. The new run stops after selection so the winner can be recorded before new test evaluation. Earlier validation histories motivate the narrower design, but original test exposure is explicitly disclosed.
