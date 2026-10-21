# Assignment 2 requirements and evidence

Mapped to the original DL4STG PA2 handout, pages 1, 3 and 4-13. This checklist records the submission requirements. The report builder's final readiness check is the authority for missing report content and assets.

## Question 1

The report must use the six headings in this order: architecture diagram, assumptions, forecasting task and data specification, model computations and justification, operation and exceptional conditions, and uncertainty/limitations/information requests. The design must forecast S1-S4 one hour ahead every ten minutes and use only information available at issuance.

Evidence: `report/q1.md`, `report/q1_arch.dot`, `report/q1_arch.png` and `report/q1_graph.png`. The proposal includes source alignment, tensor dimensions, numbered assumptions, equations and objective, graph/temporal reach, missing and delayed data, schedule-versus-actual releases, and uncertainty. Q1 requires no implementation or training. The companion notebook is optional and ungraded. There is no Q1 word or page limit.

## Question 2

| Handout item | Required evidence | Repository location |
|---|---|---|
| Setup and Table 1 | Shared transformations and window functions, synthetic checks | `pa2/data.py`, `tests/test_windows.py`, `results/table1.json` |
| Figure 1 and Q2.1-Q2.2 | Training-video flow distributions and their interpretation | `figures/fig1_flow.png`, `results/fig1_flow_stats.json`, `report/q2_part1.md` |
| Model A and Q2.3 | Exact baseline, parameter count and one-step reach | `pa2/models.py`, `logs/reach_A.txt`, `report/q2_part1.md` |
| Table 2, Figure 2, Q2.4 | Both-seed normal-speed errors and seed-0 examples | `results/table2.md`, `figures/fig2_*.png`, `report/manual.json` |
| Q2.5 | Original prediction committed before baseline stress tests | `answers/Q2_5.md`, commit `2513a26`, `logs/phase2_gate.txt` |
| Table 3, Figures 3-4, Q2.6-Q2.8 | Mirror/rotation consistency alongside correctness, worst cases and histograms | `results/table3.md`, `results/fig3_videos.json`, `figures/fig3_A*.png`, `figures/fig4_consistency_hist.png`, `gifs/fig3_A*.gif` |
| Table 4, Figure 5, Q2.9-Q2.10 | Speed failures and a controlled experiment proposal | `results/table4.md`, `results/fig5_video.json`, `figures/fig5_A_speeds.png`, `report/manual.json` |
| Q2.11 | Original prediction, at most three sentences, committed before B/C/D training | `answers/Q2_11.md`, commit `fefd535`, `logs/phase3_gate.txt` |
| Model D selection and Q2.12 | Larger reach, parameter limit, validation-only design choice | `results/d_selection.json`, `results/d_selection.md`, `logs/reach_D.txt`, `report/manual.json` |
| Table 5 and Q2.13-Q2.14 | A-D comparisons, failure attribution, accuracy/cost tradeoffs | `results/table5.md`, `results/eval/*.json`, `report/manual.json` |
| Exact-symmetry argument and Q2.15 | Group averaging proof, learned versus built-in symmetry | `pa2/models.py`, `report/q2_part3.md`, `report/manual.json` |
| Table 6 and Figure 6 | Wrapped B and D, final D-based model, same videos as Figures 3/5 | `results/table6.md`, `results/fig6.json`, `figures/fig6*.png`, `gifs/fig6a*.gif` |
| Q2.16-Q2.17 | Revisit the Q2.7 case and give a verdict of at most 150 words | `report/manual.json` |

Protocol: complete-video split 8000/1000/1000 by index, float32 intensities in [0,1], nine-step rollout with differentiable feedback and no teacher forcing, masked equal-video loss, Adam 0.001, batch 32, 30 epochs, seeds 0 and 1, final weights. B changes symmetry exposure, C changes speed exposure, and D changes architecture relative to C. All four D candidates use both seeds and 30 epochs; selection averages validation E(2) over all three speeds and both seeds and reuses both winner runs. The optional D+B decision must cite Table 5 evidence.

The fourth candidate, D_dil64, was an additional adaptive experiment after the original test analysis. Its plan was committed at `8debfef`, and its completed validation selection at `f4c92ad` preceded its test evaluation. Evidence: `experiments/d_refinement_plan.md`, `results/d_refinement_selection_record.json`, `logs/refined_evaluation_gate.json`. The original twelve-run submission is preserved at `995521b`; the current submission has fourteen runs and explicitly discloses the later experiment.

Report learned values as mean (seed 0, seed 1), errors to three significant figures and scientific notation below 0.001. Wrapped consistency gaps must use scientific notation. Use seed 0 for illustrations. Always pair a consistency gap with its transformed error. Performance examples and errors use the test split except the stated synthetic checks, training flow analysis and explicitly labelled validation monitoring/design selection.

## Final submission gates

- Submit one PDF through the LMS with all Q1 headings, Q2.1-Q2.17, Tables 1-6, Figures 1-6, captions and GIF paths.
- Put the actual GitHub repository link on page 1 and make the repository public at submission time. Publication waits until local work is ready.
- Include code, final checkpoints, measurements, training logs and GIFs. The large dataset and local virtual environment are excluded; the verified download script and dependency lists support reproduction.
- Disclose AI prompts, corresponding outputs and edits honestly. The disclosure includes available verbatim requests and nine retrospective prompts labelled as reconstructed, with outcomes and edits. The original prompt record for the initial assistance has not been supplied. The final guard explicitly retains this unresolved requirement; reconstructed prompts do not replace the historical record.
- Preserve the prediction commits and phase logs. The report retrieves the original committed answers, rather than a later rewrite.
- Retain both seed outcomes, including weak runs. The D_k5 seed-1 sleep/wake interruption is recorded in `results/sleep_interruption.json` and disclosed in the report's timing context.
- Run `python report/build.py --final` and inspect the rendered PDF pages before submission. A successful build is insufficient without visual inspection and a real public repository link.

The deadline in the original handout is 26 October 2026 at 11:55 pm. The student remains responsible for checking the report and making the LMS submission.
