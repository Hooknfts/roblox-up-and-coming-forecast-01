# Method and backtest

`main.py` is one self-contained Python file (standard library only). It reads
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json`. No network, no randomness, no clock: two runs on
the same history write the same file byte for byte (verified on the kit
input, 0.6 s a run against the 300 s limit).

## What the chart does (measured on the public archive)

The model was built from the public archive
[Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots)
(MIT), 50 reads of the chart from 2026-10-30 to 2026-11-12, plus the
practice history in the kit (the same 34 reads to 2026-11-07). Creation
dates for the analysis only came from Roblox's public games API. None of
the archive data is baked into `main.py`: the program fits everything from
whatever history it is given, so the same code runs on the final history.

- The chart churns fast: median rank move 3 places in 6 hours, 16 in
  72 hours; the top 10 keeps ~80% of its games after 12 hours, ~35%
  after 72 hours.
- Player-count growth does **not** predict rank moves (correlation
  -0.005 between log playing change and rank change), so the forecaster
  ignores player counts. Cross-chart presence (Top Trending / Top
  Playing Now) and `UPD`/`NEW` in a name were tested the same way and
  carry no usable signal either.
- Rank velocity persists a little (correlation ~0.22 between consecutive
  velocity estimates): recent climbers keep climbing, recent fallers keep
  falling.
- Games disappear at a rate that jumps past ~28 days old (median age at
  disappearance 28 days; ranks 11-30 essentially never leave before
  28 days, then leave at ~6% a read). The 28-day rule is fuzzy in
  `created` terms — many games are still on the chart past
  `created` + 28 days — so it is modelled as a gentle extra hazard, not
  a cliff.
- A quarter of the top-10 slots 72 hours out are held by games that were
  not on the chart when the history ended, ~90% of them never seen
  before. They can only be covered by the unnamed share.

## The model (see the docstring in `main.py` for the full description)

1. **Empirical rank-transition kernel.** Every pair of reads of the same
   game up to 80 hours apart gives one (start rank, gap, destination
   rank) observation, pooled by start-rank band and gap bucket. That is
   the distribution of where a game goes — drift, spread, asymmetry
   and exits included, with no shape assumed. (A Gaussian fitted to the
   same pairs overstates the tails: it would give a rank-40 game a 6%
   chance of #1 in 72 hours; the data say well under 1%.) The forecast
   is the kernel's destination distribution averaged over the window's
   hours, which is the honest probability for a log score averaged over
   the window's reads.
2. **Momentum.** The kernel is shifted by a shrunk recent velocity
   (least-squares slope of rank over the last 48 h, beta 0.10, shift
   capped at 5 places).
3. **Ageing out.** A small extra exit hazard past ~26 days old, by rank
   band (~0.75x survival over the window for an old top-10 game).
   The age is exact when `created` is in the history's `details`, else
   first-seen plus the median 21 days a game is already old when it
   first appears on the chart.
4. **Occupancy prior.** A 10% blend of the time-decayed frequency of
   each game at each place (1-day half-life): what a game has been doing
   very recently.
5. **Unnamed reserve.** 0.35 of every place is left unnamed, for games
   that have not appeared yet. A game is only named at a place when its
   probability beats what it would get unnamed (0.35 divided by the
   games left unnamed). The top-10 guess is a greedy maximum-weight
   assignment of games to places on those probabilities.

## Backtest (local scorer, validated: it reproduces the starter's official
## practice score of -43.512039 exactly)

History cut off at a date, forecast, scored on the archive reads in the
next 72 hours. The practice row (cutoff 2026-11-07T23:00Z) is the
competition's official practice window.

| cutoff | this entry (v2) | v1 | starter baseline | reads |
|---|---|---|---|---|
| 2026-11-03 | -40.31 | -42.18 | -50.9 | 10 |
| 2026-11-04 | -38.56 | -38.42 | -48.6 | 11 |
| 2026-11-05 | -45.34 | -46.95 | -55.9 | 11 |
| 2026-11-06 | -43.33 | -43.58 | -52.4 | 13 |
| 2026-11-07 (practice) | -34.70 | -35.68 | -43.51 | 10 |
| 2026-11-10 | -34.63 | -34.59 | -45.0 | 6 |
| **mean** | **-39.48** | **-40.23** | **-49.38** | |

Running the program itself on the kit's practice input scores
**-34.620** (6.4 top-10 hits, 54.7 agreement), against the house
baseline's -43.512 and the house model's practice score of -37.2.
The backtest's practice row (-34.70) differs slightly because it
injects `created` dates from the games API into the history; the
shipped program reads them only from the input's `details` when
present, which the kit's practice input does not have.

## What changed in v2 (and what was tried and rejected)

v2 is v1 with one change: the kernel's Dirichlet smoothing mass
`SMOOTH_ALPHA` 20 -> 100. The per-(band, bucket) cells are thin (a top
band has only a few dozen pairs per bucket), so the raw empirical
distribution is noisy; stronger shrinkage toward the pooled +
neighbour-band prior calibrates it. The choice was made on the
6-window cross-window backtest, not on the practice window alone:
alpha 90, 100, 110 and 120 all beat v1 (backtest mean -39.6 to -39.5,
kit practice -34.59 to -34.68), while alpha 20 is -40.23 / -35.586 and
alpha 160 falls back to -40.39. 100 is the middle of that plateau, not
a tuned peak; neighbouring values (90-120) all improve, and v2 wins
4 of the 6 backtest windows with the worst loss -0.14 (Nov 4).

Tried and **not** adopted, because the same backtest says they do not
help on the cross-window mean (all scores are that backtest's mean;
v1 is -40.23):

- **Velocity-conditional kernel** (5 classes: extreme / strong /
  mild climber / flat / faller, from the 48 h rank slope). The signal
  is real in isolation — strong climbers starting at ranks 11-70 reach
  the top 10 in 40-80 h 14% of the time, extreme climbers 26%, flat
  games 5% — and it fixes place 1 dramatically (practice-window
  place-1 score -5.2 -> -3.9). But it sharpens the top games'
  probabilities so much that the per-place 0.65 named-mass cap fills
  with climbers and crowds out the steady games that actually occupy
  places 4-10; net mean -39.9 to -40.1 (blended with the pooled kernel
  at weight 0.5), worse than the smoothing change alone. Rejected.
- **Exit counts in the kernel** (one exit observation at the first
  read where a game is absent, so the kernel is unconditional rather
  than conditioned on survival). Principled, but measured neutral
  (-40.07 at alpha 20 vs -40.23; -39.33 at the tuned constants vs
  -39.31 without it). Not adopted, to keep the shipped program the
  validated one.
- **Per-place unnamed reserve** (more reserve at place 1, less at
  places 4-10). No gain (best -39.36 vs -39.31); negative slopes hurt
  the Nov 10 window badly. Rejected.
- **Loosening the named-mass cap** (0.65 -> 0.75/0.85/0.95). Hurts a
  lot (-40.80 / -41.65 / -42.78): the unnamed pool is insurance for
  occupants the model never names, and the log score punishes a thin
  unnamed pool brutally. Rejected; the 0.35 reserve stands.
- Parameter sweeps that did not survive: momentum beta 0.2-0.3,
  occupancy 0.2-0.3, temperature 0.7-0.9 or 1.1-1.4, shorter pair gaps,
  24 h velocity windows. The full grid is in the build transcript.

## Honest caveats

- The practice leaderboard is topped by entries scoring -13.2 with
  "fitted place-occupancy tables". Those fit the public practice
  answer itself (the 10 archive reads of 2026-11-08 to 2026-11-10 are
  public), which is possible for the practice phase only. Their number
  does not carry over to the final window, whose reads do not exist
  yet. This entry does not do that: -34.62 (v2; v1 was -35.59) is a
  genuine forecast's score on the same window.
- The prize bar is the best house entry **on the final window**
  (baseline or house model, whichever is higher), which is not known
  until the window is read. This entry beats both house entries on the
  practice window (+8.9 over the baseline, +2.6 over the house model),
  but its cross-window mean (-39.5) is still below the house model's
  practice score (-37.2): whether it beats the house on the final
  window depends on the final window's churn, and is not guaranteed.
  The Nov 5/6 backtest windows show how far a regime change can pull
  any forecast down. v2 improves on v1 by +0.75 on the cross-window
  mean and +0.97 on the kit practice score, measured the same way for
  both versions; that is a real but modest improvement, and the final
  window is a fresh draw.
- The forecast is a single static distribution for all 72 hours: it
  cannot adapt inside the window, and it cannot name games published
  after the history ends.
