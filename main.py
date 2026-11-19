"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01), Python 3.13 stdlib only.

Reads one case (makefun/gta6-trophies-case/v1) from the input directory and writes
/output/<id>.out: for each trophy, five quantiles (5/25/50/75/95%) of its earn rate
on Exophase at the case's target. Deterministic: no randomness, no network, no clock.

Method
------
The rate a case asks for is a ratio: earners / players-tracked, both of which move
after launch, and Exophase only refreshes its page on its own schedule, so the
history is a step function: long flat stretches, then jumps at refreshes.

1.  Epochs. Consecutive reads with identical players-tracked and rates collapse to
    one epoch (the first read's slot). Fitting on epochs, not raw reads, keeps a
    6-hour stale stretch from counting as 36 observations.

2.  Trend (when the history moves). Trophy unlocks follow an adoption curve, so
    logit(rate) is fitted per trophy as a straight line in ln(1 + hours since
    launch) over the last 24 hours of epochs -- the log-logistic adoption form.
    The slope is shrunk toward the cross-trophy median slope (partial pooling:
    a trophy's own slope is noisy when its rate is a handful of players, so it
    borrows strength from the other 54 trophies). The fit is recency-weighted
    (8-hour half-life over a 24-hour window), so it follows the recent slope
    -- the curve's shape drifts as the launch matures -- while older epochs
    still steady it. The level is anchored on the last read, so the forecast
    is last-rate plus the shrunk slope times the extrapolation gap in
    log-time. (A separate earners-over-players-tracked view was tried and
    dropped: on synthetic launch weeks it scored worse than the rate-only
    fit, because rounded rates make small earner counts too noisy to
    extrapolate on their own.)

3.  Guards the trend alone would miss.
      * Earned-count floor. Earners never fall, so rate(target) >=
        rate(last) * N(last) / N(target), with N(target) from a log-log fit of
        players-tracked. A trend that dives below what the players who already
        earned the trophy imply is lifted to the floor.
      * Story chain. Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final
        mission >= 100% completion >= platinum, and the platinum is below every
        trophy (it needs them all). Pool-adjacent-violators on the logit medians
        enforces the order.
      * Cold start. A trophy with (almost) no earners yet gets a small tier
        prior instead of a flat zero.

4.  Flat history (the launch-night stale page). When every read is identical
    the page has not refreshed yet, and the next refresh dilutes every rate:
    Exophase's first refresh took players-tracked from 638 to 3,868 (the kit's
    own rules), a log dilution of ln(3868/638) = 1.80. So the median is the
    last rate shifted down by 1.80 times a dilution exposure per trophy --
    1.0 for late/grind trophies whose earners barely grow in the first hours,
    ~0.45-0.65 for early chapter trophies new players reach quickly, 0 for a
    trophy already at ~100% -- with a wide spread (sigma 0.9 in logit) because
    a refresh moves rates by far more than any 90-minute random walk.

5.  Spread. sigma^2 = base diffusion (0.35 * sqrt(horizon in days)) +
    slope uncertainty (shrunk slope standard error times the extrapolation
    gap) + fit residual + 0.2 for the target read's own refresh staleness
    (the answer is the page as of its last refresh before the target, which
    may be hours old). Floored at 0.4, capped at 2.0. Quantiles are the
    logistic of logit(median) + z_tau * sigma, rounded to 4 decimals half-up.

What is modelled and what is observed: every number above is computed from the
case's own history. No Exophase figures are invented: the only outside facts
are the launch date (the case's release_date) and the first refresh's
players-tracked (638 -> 3,868), both printed in the kit's rules, used only for
the flat-history fallback. Nothing here claims a rate that was not read.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. Outside the grader, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is shipped; this is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tuning constants (chosen on synthetic launch-week data, see README) ---
DILUTION = 1.80            # ln(3868 / 638): the first refresh's log dilution
FLAT_SIGMA = 0.9           # logit spread when the history is one flat page
BASE_SIGMA_PER_DAY = 0.35  # random-walk diffusion, logit per sqrt(day)
REFRESH_SIGMA = 0.2        # the target read's own refresh staleness
SIGMA_FLOOR = 0.4
SIGMA_CAP = 2.0
SHRINK_K = 1.0             # slope shrinkage: w = n_eff / (n_eff + K)
SLOPE_CAP = 2.0            # |shrunk slope| cap, logit per log-hour
FIT_WINDOW_H = 24.0        # hours of epochs the trend is fitted on
WEIGHT_HALF_LIFE_H = 8.0   # recency weight half-life within the window
N_WINDOW_H = 48.0          # hours of epochs the players-tracked fit uses


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ols(xs, ys, ws=None):
    """Least squares y = a + b x (optionally weighted); returns
    (a, b, residual_sd, se_b, n_eff)."""
    if ws is None:
        ws = [1.0] * len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    n_eff = sw * sw / sum(w * w for w in ws) if ws else 0.0
    if len(xs) > 2 and sxx > 0:
        resid = [(y - (a + b * x)) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw)
        se_b = sd / math.sqrt(sxx)
    else:
        sd = 0.0
        se_b = 0.0
    return a, b, sd, se_b, n_eff


def dilution_exposure(trophy):
    """How much of the refresh dilution a trophy feels (0 = none, 1 = all).

    A trophy new players reach in their first hours keeps earning as the
    tracked population grows, so its rate is diluted less; a late or grindy
    trophy's earners barely grow in the first hours, so it feels the full
    dilution. Only used when the history is one flat page.
    """
    text = (trophy.get("description", "") + " " + trophy.get("name", "")).lower()
    tier = trophy.get("tier", "")
    if "gone fishing" in text:
        return 0.0
    if "100% completion" in text or tier == "platinum":
        return 1.0
    if "chapter 1" in text:
        return 0.45
    if "chapter 2" in text:
        return 0.65
    if "chapter 3" in text:
        return 0.80
    if "chapter 4" in text:
        return 0.90
    if "final mission" in text:
        return 0.95 if "complete the final mission." in text else 0.85
    if tier == "gold":
        return 0.95
    if tier == "silver":
        return 0.90
    for kw in ("250 stunts", "100 locations", "50 challenges", "collectible card",
               "8 mission strands", "100 vehicles", "20 different races", "10 vehicles"):
        if kw in text:
            return 0.90
    for kw in ("fishing", "arcade", "breakfast", "cocktail", "bowling",
               "meditate", "explore all"):
        if kw in text:
            return 0.55
    return 0.75


def chain_order(trophies):
    """Indices, in story order, of the chapter/final/100%/platinum chain."""
    def kind(t):
        text = (t.get("description", "") + " " + t.get("name", "")).lower()
        if "chapter 1" in text:
            return 0
        if "chapter 2" in text:
            return 1
        if "chapter 3" in text:
            return 2
        if "chapter 4" in text:
            return 3
        if "complete the final mission." in text:
            return 4
        if "100% completion" in text:
            return 5
        if t.get("tier") == "platinum":
            return 6
        return None
    order = []
    for i, t in enumerate(trophies):
        k = kind(t)
        if k is not None:
            order.append((k, i))
    order.sort()
    return [i for _, i in order]


def pava_decreasing(values):
    """Pool adjacent violators: make values non-increasing, least-squares."""
    vals = list(values)
    blocks = [[v, 1] for v in vals]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            m0, c0 = blocks[i]
            m1, c1 = blocks[i + 1]
            blocks[i:i + 2] = [[(m0 * c0 + m1 * c1) / (c0 + c1), c0 + c1]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for m, c in blocks:
        out.extend([m] * c)
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    launch = datetime.strptime(case["game"]["release_date"] + "T00:00:00Z",
                               "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    at = parse_ts(case["at"])
    t_end = (at - launch).total_seconds() / 3600.0
    t_target = t_end + case["horizon_minutes"] / 60.0
    x_target = math.log1p(max(0.0, t_target))

    # Collapse consecutive identical reads into epochs (first slot kept).
    epochs = []  # (t_hours, players_tracked, rates)
    for h in history:
        try:
            t = (parse_ts(h["slot"]) - launch).total_seconds() / 3600.0
            n = h["players_tracked"]
            rates = h["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        if epochs and epochs[-1][1] == n and epochs[-1][2] == rates:
            continue
        epochs.append((t, n, rates))
    if not epochs:
        raise ValueError("case has no usable history")

    last_t, last_n, last_rates = epochs[-1]
    x_last = math.log1p(max(0.0, last_t))

    # Flat history: one epoch, or no trophy moved at all.
    def rates_of(ep):
        return ep[2]
    moved = False
    for ep in epochs[1:]:
        for a, b in zip(rates_of(ep), rates_of(epochs[0])):
            if a != b:
                moved = True
                break
        if moved:
            break
    flat = len(epochs) <= 1 or not moved

    medians = [0.0] * len(trophies)   # logit medians
    sigmas = [0.0] * len(trophies)

    if flat:
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = 0.0
            if r >= 99.5:
                medians[i] = logit(99.99)
                sigmas[i] = 0.05
            else:
                medians[i] = logit(r) - DILUTION * dilution_exposure(tr)
                sigmas[i] = FLAT_SIGMA
        return build_rows(trophies, medians, sigmas)

    # --- players-tracked at the target (log-log fit on recent epochs) ---
    n_fit = [(t, n) for (t, n, _) in epochs if t >= t_end - N_WINDOW_H and n > 0]
    if len(n_fit) < 3:
        n_fit = [(t, n) for (t, n, _) in epochs if n > 0]
    if len(n_fit) >= 2:
        a_n, b_n, _, _, _ = ols([math.log1p(max(0.0, t)) for t, _ in n_fit],
                                [math.log(n) for _, n in n_fit])
        n_target = math.exp(a_n + b_n * x_target)
    else:
        n_target = float(last_n)
    n_target = max(n_target, float(last_n))

    # --- per-trophy trend fits: logit(rate) linear in ln(1 + t), recency
    # weighted (half-life WEIGHT_HALF_LIFE_H) over the last FIT_WINDOW_H
    # hours of epochs. The weights let the fit follow the recent slope --
    # the curve's shape drifts as the launch matures -- while the older
    # epochs still steady it.
    fit_pts = []
    for i in range(len(trophies)):
        pts = []
        for (t, _, rates) in epochs:
            if t >= t_end - FIT_WINDOW_H and rates[i] is not None:
                wgt = 0.5 ** ((t_end - t) / WEIGHT_HALF_LIFE_H)
                pts.append((math.log1p(max(0.0, t)), logit(rates[i]), wgt))
        if len(pts) < 3:
            pts = []
            for (t, _, rates) in epochs:
                if rates[i] is not None:
                    wgt = 0.5 ** ((t_end - t) / WEIGHT_HALF_LIFE_H)
                    pts.append((math.log1p(max(0.0, t)), logit(rates[i]), wgt))
        fit_pts.append(pts)

    slopes = []
    for pts in fit_pts:
        if len(pts) >= 3:
            _, b, _, _, _ = ols([p[0] for p in pts], [p[1] for p in pts],
                                [p[2] for p in pts])
            slopes.append(b)
    b_med = median(slopes) if slopes else 0.0

    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        pts = fit_pts[i]
        if r_last is None:
            # Never seen: small tier prior, wide spread.
            prior = {"platinum": 0.02, "gold": 0.05, "silver": 0.05}.get(tr.get("tier"), 0.10)
            medians[i] = logit(prior)
            sigmas[i] = 1.5
            continue
        if r_last >= 99.5:
            medians[i] = logit(99.99)
            sigmas[i] = 0.1
            continue
        if r_last <= 0.01:
            # (Almost) no earners yet: a cold-start prior, not a flat zero.
            prior = {"platinum": 0.02, "gold": 0.05, "silver": 0.05}.get(tr.get("tier"), 0.10)
            medians[i] = logit(prior)
            sigmas[i] = 1.2
            continue
        if len(pts) < 3:
            medians[i] = logit(r_last)
            sigmas[i] = 1.0
            continue
        a, b, sd, se_b, n_eff = ols([p[0] for p in pts], [p[1] for p in pts],
                                [p[2] for p in pts])
        n_pts = len(pts)
        if n_pts < 3:
            # Too few points for a slope of its own: borrow the median slope
            # outright and admit wide uncertainty.
            b_star = b_med
            se_eff = 0.8
        else:
            # Shrinkage is stronger when epochs are scarce: a slope from
            # three points is mostly noise, so it leans on the cross-trophy
            # median; with plenty of epochs the trophy's own slope leads.
            k_eff = SHRINK_K if n_pts >= 5 else 4.0
            w = n_pts / (n_pts + k_eff)
            b_star = w * b + (1.0 - w) * b_med
            se_eff = w * se_b + (1.0 - w) * 0.3
        b_star = max(-SLOPE_CAP, min(SLOPE_CAP, b_star))
        dx = x_target - x_last
        lg = logit(r_last) + b_star * dx
        # Earned-count floor: earners never fall.
        floor_rate = r_last * last_n / n_target
        lg = max(lg, logit(floor_rate))
        lg = min(lg, logit(99.99))
        medians[i] = lg
        var = (BASE_SIGMA_PER_DAY ** 2) * horizon_days
        var += (se_eff * dx) ** 2
        var += (0.5 * sd) ** 2
        var += REFRESH_SIGMA ** 2
        sigmas[i] = min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(var)))

    # --- sparse-history floor for near-zero trophies ---
    # When the page has barely refreshed (few epochs, or the last refresh
    # was early), a trophy sitting at 0.02% is not "a trophy nobody will
    # earn" -- it is a trophy nobody had *yet* earned when the page last
    # moved, and by day 7 a typical bronze of its kind is at a few percent.
    # The logit scale punishes under-forecasting such a trophy far more
    # than over-forecasting it, so the median gets a tier prior floor.
    # Only when the horizon is long (>= 2 days); a near-zero rate is
    # informative over a short horizon. Story-chain trophies are excluded:
    # their order already constrains them.
    if horizon_days >= 2.0 and (len(epochs) < 6 or last_t < 24.0):
        chain_ids = set(chain_order(trophies))
        for i, tr in enumerate(trophies):
            if i in chain_ids:
                continue
            r_last = last_rates[i]
            if r_last is None or r_last >= 1.0:
                continue
            d = (tr.get("description") or "").lower()
            if "gone fishing" in d:
                continue
            prior = {"platinum": 0.15, "gold": 1.5, "silver": 1.5}.get(
                tr.get("tier"), 4.0)
            medians[i] = max(medians[i], logit(prior))
            sigmas[i] = max(sigmas[i], 1.2)

    # --- story-chain monotonicity on the logit medians ---
    order = chain_order(trophies)
    if len(order) >= 2:
        fixed = pava_decreasing([medians[i] for i in order])
        for k, i in enumerate(order):
            medians[i] = fixed[k]
    # Platinum below every trophy (it needs them all).
    plat = [i for i, tr in enumerate(trophies) if tr.get("tier") == "platinum"]
    if plat:
        cap = min(medians[i] for i in range(len(trophies)) if i not in plat)
        for i in plat:
            medians[i] = min(medians[i], cap)

    return build_rows(trophies, medians, sigmas)


def build_rows(trophies, medians, sigmas):
    rows = []
    for i, tr in enumerate(trophies):
        lg = medians[i]
        sg = max(0.0, sigmas[i])
        qs = []
        for z in Z:
            q = logistic(lg + z * sg)
            qs.append(round4(clamp(q)))
        # Keep the row non-decreasing after rounding.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        if not case.get("history"):
            raise ValueError(f"{path.name} has no history")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
