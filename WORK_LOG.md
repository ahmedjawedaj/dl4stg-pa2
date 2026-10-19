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
4. Started a resumable HTTPS download from the official Moving MNIST host. It must pass the published size and MD5 check before torchvision uses it.
5. Added per-epoch progress JSON and a separate atomic resume checkpoint. Resume restores optimizer, window/augmentation RNG and CPU RNG, and rejects changes to seed, model, data count or runtime. Only a completed run produces the final checkpoint filename that phase scripts skip.
6. Added an isolated test that compares uninterrupted training against an interruption/resume, checks identical final weights/losses, and rejects a seed mismatch. It passed in the new environment, as did the existing window/symmetry tests. A complete synthetic one-batch MPS trainer run, including final checkpoint serialization, passed. Synthetic batches are not assignment results.
7. Table 1, hardware recording and autograd reach checks ran successfully in the intended environment. The latter confirmed A=8 px / 480,784 parameters and all three larger-reach D configurations within the limit.
8. Installed report-generation dependencies and tested a draft PDF using the existing bundled Chromium. The builder accepts `PA2_CHROMIUM_EXECUTABLE` to avoid another browser download. Missing results remain visibly marked; the draft is not submission ready.
9. Reworked the Q1 architecture graphic after visual review showed the old diagram's labels were too small at printed size. The updated single-column graphic fits one page and was rendered and inspected.

## Next actions

1. Finish environment installation; run `tests.test_windows` and `tests.test_resume`, and confirm MPS forward/backward execution.
2. Verify the dataset checksum and loader shape. Run the synthetic Table 1 checks, Figure 1 and reach check.
3. Commit the setup/reliability changes and start phase 1 with caffeinate, durable logs and a recorded process ID.
4. Inspect the first complete epoch and report a measured timing estimate, with uncertainty about the different D architectures.
5. After phase 1, inspect Table 2/Figure 2, write Q2.4, and write/commit a genuine Q2.5 before phase 2.
6. After phase 2, inspect actual failures and complete Q2.6-Q2.10; write/commit Q2.11 before phase 3.
7. Run phase 3, make the optional D+B decision from evidence, complete all remaining writing, render and verify the PDF.
8. Obtain any unavailable original Claude prompts for the final AI disclosure. Keep all assistance and checks accurately attributed.
9. Create/publish the GitHub repository only when the local work is ready, then insert its actual URL and perform final submission checks.

Ahmed explicitly answered "Yes, continue automatically". The thread follow-up `finish-assignment-2` is ACTIVE every 10 minutes and should notify only on meaningful milestones, failures, completion or required user action. It must be disabled when the authorized assignment work is complete. Keep the Mac on and this desktop app running for local continuation.

The working report build command on this Mac is the project Python running `report/build.py` with `PA2_CHROMIUM_EXECUTABLE` pointing to `/Users/ahmedjawed/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`.
