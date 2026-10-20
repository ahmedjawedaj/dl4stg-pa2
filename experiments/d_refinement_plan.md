# Additional D validation experiment

Ahmed authorized an attempt to improve model performance on 20 October 2026, after the original twelve runs and test analysis were complete. This plan is committed before either additional run. It is not an original pre-test prediction. The genuine Q2.5 and Q2.11 answers remain unchanged.

## Evidence and hypothesis

The original training and validation histories show that both C seeds improved, while the selected D_dil seed 0 plateaued near validation E(5) = 0.0379. D_dil seed 1 improved to 0.0269. The other D designs also include flat, weak runs. The initial test results have already been seen, so this extension must be disclosed as adaptive development after the original evaluation; it is not an untouched-test experiment.

The original D_dil increased hidden width from C's 64 to 78 as well as increasing dilation. A minimally changed candidate will retain C's width and parameter shapes and change only the second layer's spatial dilation. This isolates that architectural change more closely and may improve training stability. The explanation remains a hypothesis until measured.

## Fixed candidate and protocol

- Name: D_dil64.
- Patch factor 4, two ConvLSTM layers with 64 hidden channels, 3x3 kernels, dilations [1, 2] and paddings [1, 2].
- Same default initialization, zero initial states, output projection, sigmoid and differentiable nine-step rollout.
- 480,784 parameters, 12-pixel one-step reach, no attention.
- Train from scratch using seeds 0 and 1, 30 full epochs each, Adam 0.001, batch 32, final weights and exactly C's speed sampling and masked equal-video loss.
- No changes to A, B, C, earlier D checkpoints, the data split, learning rate, loss, initialization scheme or training duration. Do not discard a weak seed.

## Selection and stopping rule

Train this one additional candidate on both seeds. Use the existing selection score: validation E(2) averaged equally over 1x, 2x and 3x, then equally over seeds 0 and 1. Compare all four designs using that score only and reuse both winner runs. Do not inspect the new candidate's test forecasts or test errors before recording the selection.

This authorized attempt adds two full training runs. Do not launch another candidate automatically if it fails. If the previous winner remains best, retain it and report this additional attempt honestly. If the new candidate wins, evaluate the selected D and its symmetry wrapper, refresh the affected tables, figures, GIFs and answers, and reconsider optional D+B only from the new Table 5.

All original artifacts remain recoverable in Git at commit 995521b. The original selection is additionally preserved in results/d_selection_initial.json and results/d_selection_initial.md. Report the additional candidate, its timing, both seed outcomes and this chronology, regardless of success.
