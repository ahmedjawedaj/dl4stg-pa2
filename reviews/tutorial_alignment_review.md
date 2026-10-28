# Review against the TA tutorial slides, 21 October 2026

## Scope and verdict

Reviewed all 11 slides in q1-river-tutorial.html and all 16 slides in q2-motion-tutorial.html, including rendered diagrams and incremental slide text. Compared them with the original handout, current report, pa2/data.py and pa2/models.py. The supplied HTML files contain no embedded video, recording link or transcript. The two separate tutorial recordings have therefore not been reviewed.

The Q2 implementation and main interpretation align with the slides. The Q1 proposal was broadly aligned but needed clearer information timing, an explicit rainfall-reading assumption, complete block equations and more careful uncertainty claims. These were corrected locally. No Q2 experiment or prediction commit was changed, and no retraining is needed for this review. Nothing was committed or pushed.

Sources supplied by Ahmed:
- /Users/ahmedjawed/Downloads/q1-river-tutorial.html, SHA256 e0cc3e8845522bface8975f824afc035fdc4f6ade843b5cc512ccd7cbbb90191
- /Users/ahmedjawed/Downloads/q2-motion-tutorial.html, SHA256 22f370cf8923616fd3298db1e33f4a810874bdff147bce5c5be90b4385b7cae6
- Original handout: ../DL4STG_PA2 2/DL4STG_PA2.pdf, especially pages 2-4.

## What the tutorials expect

Q1 is a defensible proposal, not a prescribed architecture. Explain how the river determines information paths, how elapsed time changes the available evidence, and exactly how the model combines sources. State every unsupported data or physical premise as an assumption, explain why it is reasonable, and identify its consequence. The slides explicitly leave missing data, delayed readings, schedule discrepancies, limitations and complete equations to the student.

Q2 is an investigation, not just a low-error leaderboard. Understand state versus weights, within-update reach versus recurrence, distribution changes versus architectural limits, and consistency versus correctness. Compare the same videos and use controlled A-to-B, A-to-C and C-to-D comparisons before attributing causes. The illustrative digits in the tutorials are not trained-model results and must not be substituted for measured forecasts.

## Q1 mapping and corrections

| Tutorial point | Report evidence and review outcome |
|---|---|
| Slides 2 and 8: six future steps at four stations | The report already computed all six horizons but described only the final horizon as operational. It now explicitly supplies all six, keeping the one-hour endpoint required by the handout. Diagram and tensor table agree. |
| Slides 3 and 6: follow the water and count hops | Seven-node river graph, including a latent unmonitored-tributary node, with four downstream hops from reservoir to S4. The two reverse confluence edges are an explicit extra modelling assumption, not a supplied fact. |
| Slide 4: only information known at the forecast origin | Removed the t0+2-minute arrival allowance. Measurements and plans must be received by t0. Later revisions have a later as-of time and cannot be credited to the original forecast. |
| Slide 5: variable travel time and adequate history | A6 labels the 8-hour routing and 12-hour rainfall-response bounds as provisional. A7 now explicitly treats the stage-delay relationship as a hypothesis to validate, not a universal law. The 24-hour window is a design choice to revisit if historical responses exceed it. |
| Slide 7: align sources, mask unavailable future observations | Tensor axes and masks are explicit. Rainfall is allocated to bins under the newly stated A12. Future measurements remain unavailable. Known release plans and calendar features are allowed. |
| Slide 9: write every main function out | Expanded the two temporal layers, lag-specific state gates, separate downstream/reverse weights and the 64-to-32-to-3 forecast head. The reverse coefficient is a declared design prior. Verified temporal offsets cover 0 through 256 continuously. |
| Slide 10: assumption, reason, design consequence | Numbered A1-A12 table separates supplied facts. Added the meaning of rain readings and allocation across irregular intervals. Each row gives a rationale and effect. |
| Slide 11: operational exceptions and unresolved uncertainty | Separate missing/delayed/scheduled/actual-release responses remain. Corrected the emergency-response wording and clarified that wide, calibrated uncertainty bands must be validated, not presumed. |

The slide's 24-step, six-place, four-channel example is explicitly a starting point, not a required shape. Our longer history, latent node, masks, age channels and quantile outputs are additional justified choices. They require explanation, not copying the tutorial dimensions. The model remains a proposal; its latency is an operation-count estimate, not a measured benchmark.

## How to state an assumption

Use: **I assume X, because Y is unspecified or provisionally plausible; therefore I choose Z. I would revise Z if evidence contradicts X.**

Example from the revised report: assume each rain value is an accumulation since the preceding timestamp because the handout supplies timestamps but not measurement semantics. Allocate totals to 10-minute bins by interval overlap, using a stated uniform-intensity approximation. If gauge metadata instead defines rates, integrate those rates. This connects an assumption to an actual preprocessing decision.

A gauge location is a supplied fact; which catchment it represents is an assumption. River connectivity is supplied; an 8-hour maximum travel time is an assumption. Six forecast steps are an output choice consistent with the tutorial; future measured rainfall is not an admissible assumption because the problem explicitly excludes it.

## Q2 mapping

| Tutorial slides | Check against our solution |
|---|---|
| 1-3: five-frame history and learned recurrent memory | Five inputs, state reset per video, learned feature-map states and reused cell weights in pa2/models.py. |
| 4-5: folding and stacked recurrence | 4x4 pixel_unshuffle produces 16x16x16, two 64-channel baseline layers, 1x1 head, shuffle and sigmoid. No averaging in folding. |
| 6-7: rollout and error feedback | Nine updates, outputs from updates 5-9, predictions fed back from update 6 without detach. |
| 8-9: within-update reach | Longest A path uses two neighbourhood convolutions, scaled by four pixels per grid cell, giving eight pixels. Existing gradient checks and Q2.3 explain the paths. |
| 10: fair speed comparison | Exact 1x/2x/3x windows, five predictions always produced, first two horizons compared across speeds, invalid targets masked. |
| 11-12: appearance and trajectory transformations | Shared mirror/anticlockwise rotation on every input and target frame; flow histograms inform familiarity claims. |
| 13: consistency is not accuracy | Tables pair C with transformed E; Q2.15-Q2.17 retain blur, position and bounce failures despite rounding-level consistency. |
| 14: change one factor | A-to-B tests symmetry exposure; A-to-C tests speed exposure; refined C-to-D keeps width/count and changes dilation. Q2.2 now avoids saying a single stress test uniquely identifies a cause. |
| 15: eight views, inverse, average | Symmetrised implementation uses every group element and its inverse. Report gives proof, cost and original-error tradeoffs. |
| 16: trajectory evidence and missing tests | Same videos are retained, movement and bounce failures are discussed, and Q2.17 proposes unfamiliar non-digit shapes as a missing test. |

## Practical readiness

The tutorial review finds no reason to rerun Q2. Its results already support the requested analysis, including imperfect forecasts. Q1 is more detailed than the tutorial skeleton, but its optional features are now tied to explicit assumptions and equations. Remaining limitations are scientific limitations stated in the report, not reasons to invent better results. The separate recordings and any additional course-policy directions inside them remain outside this review's scope.
