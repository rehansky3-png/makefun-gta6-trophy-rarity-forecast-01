# GTA VI Trophy Rarity Forecast — week one

A forecaster for the `gta6-trophy-rarity-forecast-01` bounty: given Exophase's
public PS5 trophy page for Grand Theft Auto VI as it stands at a checkpoint
(launch night, 19 Nov 2026), predict the share of tracked players holding
each of the 55 trophies one week after launch (26 Nov 2026), as five
quantiles (5 / 25 / 50 / 75 / 95 percent) per trophy.

Program: `main.py`, Python 3.13 standard library only. No network, no clock,
no randomness — two runs on the same case are byte-identical.

## How to run

```sh
# the harness sets these; the defaults are /input and /output
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

For each `/input/<id>.in` case it writes `/output/<id>.out`, a single JSON
object `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}` with
quantiles in percent, non-decreasing, between 0.01 and 99.99. Output is
2,780 bytes per case (limit 65,536). Runtime is ~0.03 s per case (limit 8 s).

Check it on the kit's practice cases:

```sh
cp <kit>/input/p00.in <kit>/input/p01.in input/
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

## Observed vs modelled — read this first

Only two things in this entry are *observed*:

- the trophy list, tiers and descriptions, which come from the input case
  (sourced by the competition from Exophase's public trophy page);
- the history inside each input case (the page's own reads: players tracked
  and rates per trophy).

Everything else is a *model* of how those numbers move, and the code says
so wherever it matters. No Exophase numbers are invented: no fake player
counts, no fake rates, no fake refresh times. Where the history gives no
signal at all, the program says so with a wide interval and a labelled
prior instead of a confident-looking number.

The one hard fact the method leans on is in the competition's own rules:
Exophase's page was frozen from 18 Nov 23:48Z to 19 Nov 04:00Z (638 players
tracked, every rate identical), then refreshed by 04:30Z to 3,868 players
with every rate but the 100% one lower. That refresh is a *dilution* event:
the player base grew ~6× while earners grew slowly, so rates fell. The
pure-dilution log shift is ln(3868/638) = 1.80.

## Method

**1. Collapse the history into epochs.** Exophase refreshes on its own
clock; between refreshes the page repeats itself. Consecutive identical
reads are one epoch, so the fit sees refreshes, not polling noise.

**2. Flat history → dilution fallback.** If the page never moved (the two
practice cases: one flat page, 638 players), there is no trend to fit.
The median is the last rate shifted down by the 1.80 dilution, scaled by
a per-trophy *dilution exposure*: late and grind trophies take the full
shift (their earners barely grow while the base explodes), early chapter
trophies take 0.45–0.65 of it (new players reach them quickly), and a
trophy already at ~100% stays at ~100%. Spread is a fixed 0.9 logit —
wide on purpose, because a single flat page says almost nothing about
where the rates land.

**3. Otherwise → a trend, per trophy.** `logit(rate)` is fitted as linear
in `ln(1 + hours since launch)` over the last 24 hours of epochs, with
recency weights (8-hour half-life: the curve's shape drifts as the launch
matures, so the recent slope matters most). The slope is shrunk toward
the cross-trophy median slope (partial pooling: a trophy whose rate is a
handful of players has a noisy slope, so it borrows strength from the
other 54), capped at ±2 logit per log-hour, and the level is anchored on
the last read. Shrinkage is stronger when epochs are scarce. A separate
earners-over-players-tracked view was tried and dropped: on synthetic
launch weeks it scored worse than the rate-only fit, because rounded
rates make small earner counts too noisy to extrapolate on their own.

**4. Guards, because the trend can lie.**
- *Earned-count floor:* earners never fall, so the target rate is at
  least `last rate × players(last) / players(target)`, where
  players(target) comes from a log-log fit of players-tracked.
- *Story-chain monotonicity:* Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥
  Chapter 4 ≥ final mission ≥ 100% ≥ platinum, and platinum ≤ every
  trophy, enforced on the logit medians (pool-adjacent-violators). A
  trophy cannot be rarer than the trophy that requires it.
- *Cold starts:* a trophy with (almost) no earners yet gets a tier
  prior (platinum 0.02%, gold/silver 0.05%, bronze 0.10%) with a wide
  spread, not a flat zero.
- *Sparse-history floor:* when the page has barely refreshed (fewer
  than 6 epochs, or the last refresh was within 24 h of launch) and the
  horizon is long (≥ 2 days), a trophy sitting at 0.02% is not "a
  trophy nobody will earn" — it is one nobody had *yet* earned when the
  page last moved. Its median gets a tier prior floor (platinum 0.15%,
  gold/silver 1.5%, bronze 4%), because the logit score punishes
  under-forecasting such a trophy far more than over-forecasting it.
  Story-chain trophies are excluded (their order already constrains
  them).

**5. Spread.** σ² = (0.35·√horizon-days)² (random-walk diffusion) +
(slope standard error × extrapolation gap)² + (0.5·fit residual)² +
0.2² (the target read's own refresh staleness), floored at 0.4 and
capped at 2.0 logit. Quantiles are the logistic of
`logit(median) + z·σ`, z = −1.645, −0.674, 0, 0.674, 1.645.

## Validation (synthetic — not official scores)

The kit does not publish a local scorer for this competition, so the
entry was validated on synthetic launch weeks: a generator that simulates
Exophase refreshes, player adoption and per-trophy earning (log-logistic
time-to-earn), then scores the forecast with the competition's own
pinball loss on the logit scale, against the two house baselines
reconstructed from `competition.yaml` (persistence and the log-time
curve). Two ground truths were used: one matching the forecaster's
assumed form, and one deliberately misspecified (Bass-diffusion
adoption, Weibull time-to-earn), so a margin that survives both is real.

Final-like cases (checkpoint ~59.5 h after launch, target day 7),
12 fresh seeds each, mean pinball loss (lower is better):

| truth | this entry | house persistence | house curve |
|---|---|---|---|
| log-logistic | **0.29** | 0.65 | 0.41 |
| Bass / Weibull (misspecified) | **0.48** | 0.59 | 0.66 |

Stress test — Exophase refreshing only every ~12 h (few epochs):
log-logistic **0.46** vs 1.22 / 1.17; Bass/Weibull **0.55** vs 1.27 / 1.63.
The sparse-history floor is what keeps the slow-refresh case from
blowing up (without it the worst seed scored 3.6).

Practice cases (flat history): the house states its persistence baseline
at ≈ 1.51 mean pinball loss. Against a family of plausible practice
answers built from the competition rules (first refresh to 3,868
players, every rate down) plus the live Exophase page as an extra proxy,
this entry scores ≈ 0.26–0.33 (mean 0.28) versus ≈ 1.6 for persistence.

Caveats, stated plainly: all of this is synthetic validation. The real
final case is one draw, and the real data-generating process is unknown —
the misspecified-truth column is the honest guide to how much worse it
could be. The intervals are calibrated to the model's own uncertainty,
not to a guarantee.

## Files

- `main.py` — the whole program (stdlib only).
- `README.md` — this file.
- `LICENSE` — MIT.
- `.makefun/owner.json` — entry ownership (wallet).

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rehansky3-png/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rehansky3-png/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
