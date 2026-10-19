### Q2.4

<div class="answer">

{{ANSWER_Q2_4}}

</div>

### Q2.5 (prediction, written and committed before any test in Section 3)

<div class="prediction">

{{ANSWER_Q2_5_COMMITTED}}

</div>

<p class="caption">Inserted by the report build directly from the git commit recorded in <code>logs/phase2_gate.txt</code> (<code>answers/Q2_5.md</code>), committed at {{Q2_5_COMMIT}}. The stress tests started at {{PHASE2_START}} (<code>logs/phase2_gate.txt</code>).</p>

## 3. Testing the Baseline

### Mirror and rotation

**Table 3.** Error at 1×, consistency gaps and transformed errors, $t = 1$ (`scripts/phase2_stress.py`).

{{table3}}

![](../figures/fig3_A.png)

<p class="caption"><b>Figure 3.</b> The two test videos with the largest per-video C<sub>M</sub> and the two with the largest C<sub>R</sub> (model A, seed 0). Rows: (1) X | f(X), (2) T(X) | T(Y), (3) T(X) | f(T(X)), (4) T(X) | T(f(X)). GIFs stepping through the five horizons: {{FIG3_GIFS}}.</p>

![](../figures/fig4_consistency_hist.png)

<p class="caption"><b>Figure 4.</b> Per-video C<sub>M</sub> (a) and C<sub>R</sub> (b) of model A (seed 0) over the 1,000 test videos, log scale. Orange lines mark the Figure 3 videos.</p>

### Q2.6

<div class="answer">

{{ANSWER_Q2_6}}

</div>

### Q2.7

<div class="answer">

{{ANSWER_Q2_7}}

</div>

### Q2.8

<div class="answer">

{{ANSWER_Q2_8}}

</div>

### Faster motion

**Table 4.** Model A and Copy-last across speeds, $t = 1$. Only valid targets are scored, so the 3× column uses the two existing targets (frames 16 and 19).

{{table4}}

![](../figures/fig5_A_speeds.png)

<p class="caption"><b>Figure 5.</b> Model A (seed 0) on one test video whose digits do not overlap in any displayed true frame: five input frames, the first two predictions and the first two true frames at 1×, 2× and 3×. {{FIG5_CAPTION}}</p>

### Q2.9

<div class="answer">

{{ANSWER_Q2_9}}

</div>

### Q2.10

<div class="answer">

{{ANSWER_Q2_10}}

</div>
