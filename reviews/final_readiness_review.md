# Final readiness review

## Disclosure revision, 21 October

At Ahmed's request, the disclosure now includes nine condensed reconstructed review prompts and recorded outcomes, alongside available verbatim requests and accurate tool attribution. The revised disclosure is two pages and the report is 27 pages. Pages 1-25 are unchanged except total-page footers. Both new disclosure pages were rendered and visually checked. The original prompt record for the initial assistance remains unsupplied, so the strict final guard still records that gap. This revision is not a recovered historical transcript or a certification of full disclosure compliance.


## Update after the 20 October refinement

The original review below describes the twelve-run submission at `995521b`. A later authorized candidate, D_dil64, completed both required seeds and was selected on validation before its test evaluation. There are now fourteen full runs. The new final model's normal-speed test E(5) is 0.027208 versus the original final model's 0.032845, a 17.2% reduction. Its seed-0 forecasts are localized moving patches instead of the earlier near-uniform field, but remain blurred and imperfect through overlap and bounces. Selected D's mean 3x E(2) is 2.11% better than C, while its 1x E(2) is worse. The earlier conclusion that selected D is worse than C at every speed no longer applies.

The report explicitly discloses the adaptive extension after original test analysis and retains every previous run and prediction commit. Original Claude prompts remain deferred. See `experiments/d_refinement_plan.md`, `results/d_refinement_selection_record.json` and the updated report for the current evidence.

## Original review, 19 October

Reviewed against the original handout and the completed local submission on 19 October 2026.

## Verdict

The experiment and its analysis are complete. The submission is still a draft because the original Claude prompts are explicitly deferred to Ahmed. The final build correctly refuses that one missing field. No LMS submission has been made.

## Report length

Twenty-six pages are not required. The handout explicitly gives Q1 no word or page limit, while asking for direct and concise answers. There is no overall report page limit stated. The current length is permissible, but Q1 prose and generous figure/table pagination could be tightened. Length is an editorial issue, not an unmet requirement. Do not cut required cases, tables, figures or the original prediction answers merely to reduce the count.

## Compliance and implementation

- Q1 has the six required headings in order, its diagram, assumptions, tensors, causal availability rules, equations, operation exceptions and uncertainty treatment. Q1 implementation and benchmarking are not required.
- Q2.1-Q2.17, Tables 1-6, Figures 1-6, captions and eight animated GIFs are present. Q2.11 and Q2.17 satisfy their length limits.
- The split, window ranges, differentiable nine-step rollout, masked equal-video loss, optimiser, batch size, 30 epochs, final weights and seeds 0/1 match the protocol.
- A has 480,784 parameters and 8-pixel reach. Selected D has 703,888 parameters and 12-pixel reach, below the cap and without attention. Candidate selection used validation results only and reused the winner runs.
- The twelve runs come from the chosen three-candidate D screening plan. The handout does not require three D candidates; screening them is additional validation work.
- Both original prediction commits precede the corresponding experiments. They remain unedited. No retrospective predictions or measurements were invented.
- Stored summaries match the saved 1,000-video metric arrays. The optional D+B decision matches the recorded Table 5 hash.
- The report has the public repository link on page 1. Prior full-page visual review and the PDF hash remain valid because this review did not edit report sources or regenerate its PDF.

## Scientific limitation

D seed 0 learned a weak, nearly featureless forecast. D's two-seed mean is worse than C, and wrapping D guarantees consistency without recovering accurate motion. This is explicitly reported rather than hidden. The handout gives no numerical accuracy pass threshold, but this result limits how strongly the final model can be defended. The report must not claim that larger reach reliably fixed the remaining failure or that exact consistency proves motion learning.

## Engineering and fixes

The ConvLSTM implementation is small and follows the specified computation. The extra machinery is mostly experiment recovery, provenance and report assembly. Those are useful for this workload; a separate application or additional architecture framework is unnecessary.

This review found and fixed a real execution-wrapper bug: run_all.sh waited for logs/db_decision.txt, while the actual Table 5 decision is results/db_decision.json. It now reads the JSON boolean and rejects an invalid value. Completed experiments were launched through phase scripts and were unaffected. README commands now consistently use the virtual environment and explain checkpoint reuse, separate-clone reproduction, retraining, report dependencies and the remaining draft field.

Verification after the fix: shell syntax checks, the existing window/transformation checks, the interruption/resume regression check, and isolated wrapper runs for false, true and invalid decision values all passed. The twelve full training logs, parameter/reach limits, stored metric summaries, validation choice and Table 5 decision were rechecked. No training or evaluation outputs were overwritten.

## Remaining submission work

Ahmed will add the original Claude prompt disclosure later, as requested. Then rebuild with --final, inspect any changed pages, and upload the PDF through the LMS. Shortening the report is optional. Understand the architecture, evidence and weak D outcome before submission or a possible viva.
