# Spectrum Management AutoResearch Program

## Goal
Improve spectral efficiency while preserving fairness and a sub-10 ms allocation decision target.

## Constraints
- Do not modify source files automatically.
- Record each trial in `results.tsv` and summarize the outcome in `learnings.md`.
- Submit any GPU training experiment through Slurm.
- Keep each trial bounded by its configured time budget.

## Candidate Hypotheses
1. Classical water-filling is best for nominal channels.
2. Adversarial water-filling improves resilience under jamming.
3. DIFFRACT reduces allocation latency with competitive capacity.
4. The hybrid agent selects the appropriate allocation strategy.