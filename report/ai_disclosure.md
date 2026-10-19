# AI-use disclosure

**Tools.** Claude (Anthropic), used as an agent with access to a cloud workspace and to my assignment folder. A second AI assistant independently reviewed the implementation against the handout. Its reviews are in `reviews/` in the repository. Training and evaluation ran on my own machine ({{HARDWARE}}).

**What Claude produced.** It read the handout and the supplied folder, wrote the Q2 code (`pa2/`, `scripts/`, the three phase scripts), and tested it on synthetic inputs and untrained models (Table 1 checks, parameter count, window and symmetry tests, smoke runs). It also produced the tables and figures from my training runs, drafted the Q1 proposal and the Q2 answers, and wrote the report build script.

**Prompts I gave (verbatim, in order).**

{{AI_PROMPTS}}

**My edits and checks.**

{{AI_EDITS}}

**Ordering of the prediction answers.** The report build inserts Q2.5 and Q2.11 from the git commits recorded when the corresponding phase started (`logs/phase2_gate.txt`, `logs/phase3_gate.txt`), and flags any later difference in the working copy. `run_phase2.sh` and `run_phase3.sh` refuse to start until the answer file is committed without uncommitted edits.
