# AI-use disclosure

**Tools and roles.** Claude (Anthropic) prepared the initial implementation and report drafts. Codex (OpenAI) independently reviewed the implementation against the handout and then took over local execution at my request. Its reviews are in `reviews/` and the execution record is in `WORK_LOG.md`. Reported measurements are produced locally on {{HARDWARE}}.

**Outputs and responsibility.** Claude supplied the initial Q2 code, experiment scripts, Q1 proposal, partial Q2 prose and report builder, and revised them after the independent reviews. Codex verified the checks in the actual environment, corrected local setup, added tested recovery for interrupted runs, ran the experiments, analysed the results and inspected the PDF layout. Claude stopped editing before training. Numerical claims come from recorded experiment outputs. This disclosure does not claim that I personally performed automated checks or inspected every result.

**User prompts and corresponding outputs.**

{{AI_PROMPTS}}

**Edits to generated work and verification.**

{{AI_EDITS}}

**Ordering of the prediction answers.** The report build inserts Q2.5 and Q2.11 from the git commits recorded when the corresponding phase started (`logs/phase2_gate.txt`, `logs/phase3_gate.txt`), and flags any later difference in the working copy. `run_phase2.sh` and `run_phase3.sh` refuse to start until the answer file is committed without uncommitted edits.
