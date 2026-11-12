#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun bounty
roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of the chart: each game's rank and
player count) plus /input/task.json, and writes /output/forecast.json: for
each of the top 10 places, a probability for each game named being at that
place, and the guess at the top 10 in order. One forecast, scored against
every hourly read of the 72 hours after the history ends.

Method (empirical rank-transition kernel, fitted from the history itself)
-------------------------------------------------------------------------
The chart churns fast: from the public archive's reads, the median rank
move is 3 places in 6 hours and 16 places in 72 hours, the top 10 keeps
only about 35% of its games after 72 hours, and a quarter of the top-10
slots 72 hours out are held by games that were not on the chart when the
history ended. Player-count growth does not predict rank moves
(correlation ~0.00), so this forecaster ignores it. What does predict is:

1. The rank-transition kernel.  Every pair of reads of the same game up
   to 80 hours apart contributes one (start rank, gap, destination rank)
   observation. Pooled by start-rank band and gap bucket, these give an
   empirical distribution over where a game starting at a given rank ends
   up after a given gap -- drift, spread, asymmetry and exits included,
   with no shape assumed. (A Gaussian fitted to the same pairs
   overstates the tails: it would give a rank-40 game a 6% chance of #1
   in 72 hours, where the data say well under 1%.) The forecast for a
   game is that kernel's destination distribution, averaged over the
   window's hours. Averaging over the hours is the right target: the log
   score averages over the window's reads, so the honest probability is
   the time-average of the occupancy probability.

2. Momentum.  A game's rank velocity over the last two days persists
   (correlation ~0.22 between consecutive velocity estimates), so the
   kernel's destination distribution is shifted by a shrunk velocity:
   beta * v * h places. The velocity is the least-squares slope of the
   game's rank against time over its recent reads.

3. Ageing out.  Only games published in the last 28 days are on the
   chart, and games disappear at a rate that jumps once they pass about
   28 days old (measured on the public archive with the games API's
   `created` dates: the median age at disappearance is 28 days, and
   games in ranks 11-30 essentially never leave before 28 days, then
   leave at ~6% a read). The kernel's exit bin already holds the average
   exit rate, so on top of it each game carries a small extra hazard
   once it passes ~26 days old, by rank band -- about 0.75x survival
   over the window for an old top-10 game, a gentle correction, not a
   wipe-out: a game still on the chart past 28 days is usually still
   there because its `created` date is older than whatever Roblox
   counts. The age is exact when the creation date is known (from the
   history's `details`), or estimated from the first time the game was
   seen on the chart plus the median 21 days a game is already old
   when it first appears.

4. Occupancy prior.  A small blend (10%) of the time-decayed frequency
   of each game at each place (1-day half-life): what a game has been
   doing very recently. A game parked at one place stays likelier to be
   there than the pooled kernel alone says.

5. New games.  A quarter of the top-10 slots 72 hours out are held by
   games that were not on the chart when the history ended, most of them
   never seen before, so they cannot be named at all. They are covered by
   an explicit unnamed reserve (0.35 of every place): a game is only
   named at a place when its probability there beats what it would get
   unnamed (0.35 divided by the games left unnamed), and the named mass
   is capped at 0.65. Games that left the chart but were seen before
   keep a small, widened chance of returning.

6. Calibration.  Every place's named probabilities are the model's
   honest per-place probabilities (the sharpening exponent TEMP is left
   at 1.0: backtesting showed sharpening hurts, so the honest values
   stand). The top 10 guess is a greedy maximum-weight assignment of
   games to places on those probabilities, which also serves the
   agreement tie-break.

Everything is deterministic: no randomness, no clock, no network. Two
runs on the same history write the same forecast byte for byte.

Standard library only. Run locally with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone

# --- Tunables (chosen by backtesting on the public archive; see METHOD.md) ---
MOMENTUM_BETA = 0.10      # shrink on the recent rank velocity (places/hour)
OCC_BLEND = 0.10          # weight of the empirical occupancy prior
OCC_HALF_LIFE_DAYS = 1.0  # decay of the occupancy prior (days)
TEMP = 1.00               # sharpening exponent on the probabilities (1 = honest)
UNNAMED_RESERVE = 0.35    # probability mass left unnamed at every place
# Extra exit hazard (per hour) once a game passes ~26 days old: the 28-day
# cliff is real but gentle for top games -- measured on the public archive,
# a top-10 game aged 28-35 days leaves at ~0.5% an hour against ~0.1%
# younger, i.e. about 0.75x survival over the window, not a wipe-out.
AGE_ZONE_START = 26.0     # days old: the extra hazard starts here
AGE_ZONE_FULL = 30.0      # days old: the extra hazard reaches its rate
AGE_EXTRA_HAZARD = (0.004, 0.008, 0.010)  # per hour, by band: top 10, 11-30, 31+
AGE_PRIOR_DAYS = 21.0     # median age of a game when it first appears
MAX_PAIR_GAP_HOURS = 80.0
MAX_ROWS_PER_PLACE = 60

# Start-rank bands for the transition kernel (edges, inclusive start).
BAND_EDGES = [1, 2, 3, 4, 6, 8, 11, 16, 26, 41, 71, 200]
# Gap buckets (hours) for the transition kernel.
GAP_BUCKETS = [3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
# Destination bins: exact ranks 1..10, then ranges, then off the chart.
DEST_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
             (8, 8), (9, 9), (10, 10), (11, 15), (16, 20), (21, 30),
             (31, 50), (51, 75), (76, 100), (101, 10 ** 9)]
# Dirichlet smoothing mass, spread over the neighbouring band's prior.
SMOOTH_ALPHA = 20.0

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def parse_time(text):
    """ISO 8601 UTC string -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def band_of(rank):
    """Index of the start-rank band a rank falls in."""
    for i in range(len(BAND_EDGES) - 1):
        if BAND_EDGES[i] <= rank < BAND_EDGES[i + 1]:
            return i
    return len(BAND_EDGES) - 2


def bucket_of(gap_hours):
    """Index of the gap bucket nearest (in log) to a gap in hours."""
    best, best_dist = 0, None
    for i, centre in enumerate(GAP_BUCKETS):
        dist = abs(math.log(gap_hours / centre))
        if best_dist is None or dist < best_dist:
            best, best_dist = i, dist
    return best


def dest_bin_of(rank):
    """Index of the destination bin a rank falls in (None = off chart)."""
    if rank is None:
        return len(DEST_BINS) - 1
    for i, (lo, hi) in enumerate(DEST_BINS):
        if lo <= rank <= hi:
            return i
    return len(DEST_BINS) - 1


# --------------------------------------------------------------------------
# loading the history
# --------------------------------------------------------------------------

def load_history(input_dir):
    """Return (snapshots, task, created).

    snapshots: list of {"at": datetime, "games": {uid: rank},
    "chart_size": int}, oldest first.
    task: the task.json dict (may be empty when absent).
    created: {uid: datetime} from the newest details that name it.
    """
    task = {}
    try:
        with open(os.path.join(input_dir, "task.json")) as f:
            task = json.load(f)
    except (OSError, ValueError):
        task = {}
    snapshots = []
    path = os.path.join(input_dir, "history.ndjson")
    parsed = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            snap = json.loads(line)
            at = parse_time(snap["at"])
            games = {}
            for row in snap.get("up_and_coming") or []:
                games[int(row["universe_id"])] = int(row["rank"])
            parsed.append((at, games, snap.get("details")))
    created = {}
    for _at, _games, details in reversed(parsed):
        if not details:
            continue
        for uid_text, det in details.items():
            if not det or not det.get("created"):
                continue
            uid = int(uid_text)
            if uid not in created:
                created[uid] = parse_time(det["created"])
    for at, games, _details in parsed:
        snapshots.append({"at": at, "games": games, "chart_size": len(games)})
    return snapshots, task, created


def game_series(snapshots):
    """Per-game time series: uid -> [(hours_since_first_snap, rank)]."""
    t0 = snapshots[0]["at"]
    series = {}
    for snap in snapshots:
        hours = (snap["at"] - t0).total_seconds() / 3600.0
        for uid, rank in snap["games"].items():
            series.setdefault(uid, []).append((hours, rank))
    return series, t0


# --------------------------------------------------------------------------
# fitting the transition kernel from the history
# --------------------------------------------------------------------------

def fit_kernel(snapshots, series):
    """Fit P(destination bin | start band, gap bucket) from every pair of
    reads of the same game up to MAX_PAIR_GAP_HOURS apart.

    Returns kernel(rank, hours) -> list of probabilities over DEST_BINS,
    log-interpolated between gap buckets and Dirichlet-smoothed toward
    the neighbouring bands' distributions.
    """
    n_bands = len(BAND_EDGES) - 1
    n_buckets = len(GAP_BUCKETS)
    n_bins = len(DEST_BINS)
    # counts[band][bucket][bin]
    counts = [[[0] * n_bins for _ in range(n_buckets)] for _ in range(n_bands)]
    for uid, obs in series.items():
        for i in range(len(obs)):
            t_i, r_i = obs[i]
            band = band_of(r_i)
            for j in range(i + 1, len(obs)):
                t_j, r_j = obs[j]
                gap = t_j - t_i
                if gap < 2.0:
                    continue
                if gap > MAX_PAIR_GAP_HOURS:
                    break
                counts[band][bucket_of(gap)][dest_bin_of(r_j)] += 1

    # Pooled distribution over all bands, as the smoothing prior.
    pooled = [[0.0] * n_bins for _ in range(n_buckets)]
    for k in range(n_buckets):
        total = sum(sum(counts[b][k]) for b in range(n_bands))
        for d in range(n_bins):
            pooled[k][d] = sum(counts[b][k][d] for b in range(n_bands))
        if total > 0:
            pooled[k] = [c / total for c in pooled[k]]
        else:
            pooled[k] = [1.0 / n_bins] * n_bins

    # Smoothed distribution per (band, bucket): counts + alpha * prior,
    # where the prior is a blend of the pooled distribution and the two
    # neighbouring bands' raw distributions (ranks behave alike nearby).
    dist = [[[0.0] * n_bins for _ in range(n_buckets)] for _ in range(n_bands)]
    for b in range(n_bands):
        neighbours = [bb for bb in (b - 1, b + 1) if 0 <= bb < n_bands]
        for k in range(n_buckets):
            n = sum(counts[b][k])
            prior = list(pooled[k])
            for bb in neighbours:
                nb = sum(counts[bb][k])
                if nb > 0:
                    for d in range(n_bins):
                        prior[d] += counts[bb][k][d] / nb
            prior = [p / (1.0 + len(neighbours)) for p in prior]
            denom = n + SMOOTH_ALPHA
            dist[b][k] = [(counts[b][k][d] + SMOOTH_ALPHA * prior[d]) / denom
                          for d in range(n_bins)]

    def kernel(rank, hours):
        """Distribution over destination bins after `hours`, log-interpolated
        between the fitted gap buckets (flat outside)."""
        b = band_of(rank)
        if hours <= GAP_BUCKETS[0]:
            return dist[b][0]
        if hours >= GAP_BUCKETS[-1]:
            return dist[b][-1]
        for k in range(len(GAP_BUCKETS) - 1):
            if GAP_BUCKETS[k] <= hours <= GAP_BUCKETS[k + 1]:
                t = ((math.log(hours) - math.log(GAP_BUCKETS[k]))
                     / (math.log(GAP_BUCKETS[k + 1]) - math.log(GAP_BUCKETS[k])))
                return [dist[b][k][d] * (1.0 - t) + dist[b][k + 1][d] * t
                        for d in range(n_bins)]
        return dist[b][-1]

    return kernel


def recent_velocity(obs, now_hours, window_hours=48.0):
    """Least-squares slope of rank against time (places per hour) over the
    recent reads, 0 when there are too few."""
    recent = [(t, r) for t, r in obs if now_hours - t <= window_hours]
    if len(recent) < 3:
        recent = obs[-3:]
    if len(recent) < 2:
        return 0.0
    n = len(recent)
    mt = sum(t for t, _ in recent) / n
    mr = sum(r for _, r in recent) / n
    num = sum((t - mt) * (r - mr) for t, r in recent)
    den = sum((t - mt) ** 2 for t, _ in recent)
    if den <= 0:
        return 0.0
    slope = num / den
    # A rank cannot keep moving faster than about a place an hour.
    return max(-1.0, min(1.0, slope))


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------

def _density_mass(density, a, b):
    """Mass of a density that is uniform over each integer rank's interval
    [r - 0.5, r + 0.5], inside the interval [a, b]."""
    if b < 0.5 or a > 100.5:
        return 0.0
    a = max(a, 0.5)
    b = min(b, 100.5)
    total = 0.0
    lo_r = max(1, math.floor(a + 0.5))
    hi_r = min(100, math.ceil(b - 0.5))
    for r in range(lo_r, hi_r + 1):
        seg_a = max(a, r - 0.5)
        seg_b = min(b, r + 0.5)
        if seg_b > seg_a:
            total += density[r] * (seg_b - seg_a)
    return total


def place_probs(dist, shift, places):
    """Probability of each place 1..places under the kernel distribution
    `dist` (over DEST_BINS), shifted by `shift` places (positive means
    falling to a worse rank). The kernel's mass is spread uniformly inside
    each destination bin; the off-chart bin never reaches a place."""
    density = [0.0] * 101  # ranks 1..100, indexed by rank
    for d, (lo, hi) in enumerate(DEST_BINS):
        if lo > 100:
            continue  # off the chart
        hi = min(hi, 100)
        per = dist[d] / (hi - lo + 1)
        for r in range(lo, hi + 1):
            density[r] += per
    out = []
    for k in range(1, places + 1):
        # Place k is true rank in [k - 0.5, k + 0.5]; rank 1 takes
        # everything better, so its interval starts at -infinity.
        # True rank = kernel rank + shift.
        a = -1e9 if k == 1 else k - 0.5 - shift
        b = k + 0.5 - shift
        out.append(_density_mass(density, a, b))
    return out


def build_forecast(snapshots, task, created):
    """Return the forecast dict {"probabilities": [...], "top10": [...]}."""
    places = int(task.get("places", 10))
    window_hours = int(task.get("window", {}).get("slots", 72))
    last = snapshots[-1]
    t_end = last["at"]
    close_size = last["chart_size"] or 100

    if len(snapshots) < 2 or not last["games"]:
        # Too little history to fit anything: persistence, as the starter.
        chart = sorted(last["games"], key=lambda u: last["games"][u])
        probs = []
        for place in range(1, places + 1):
            if place - 1 < len(chart):
                probs.append({"universe_id": chart[place - 1], "place": place, "p": 0.5})
            if place == 1 and len(chart) > 1:
                probs.append({"universe_id": chart[1], "place": place, "p": 0.3})
            elif place > 1:
                if place - 2 >= 0:
                    probs.append({"universe_id": chart[place - 2], "place": place, "p": 0.15})
                if place < len(chart):
                    probs.append({"universe_id": chart[place], "place": place, "p": 0.15})
        return {"probabilities": probs, "top10": chart[:places]}

    series, t0 = game_series(snapshots)
    kernel = fit_kernel(snapshots, series)
    now_hours = (t_end - t0).total_seconds() / 3600.0

    # Empirical occupancy prior: time-decayed frequency of each place.
    occ = {}
    occ_weight = {}
    for uid, obs in series.items():
        total_w = 0.0
        arr = [0.0] * (places + 1)
        for t, rank in obs:
            age_days = max(now_hours - t, 0.0) / 24.0
            w = 0.5 ** (age_days / OCC_HALF_LIFE_DAYS)
            total_w += w
            if rank <= places:
                arr[rank] += w
        if total_w > 0:
            occ[uid] = [a / total_w for a in arr]

    # Per-game, per-place probability, averaged over the window's hours.
    prob = {}
    for uid, obs in series.items():
        last_t, last_rank = obs[-1]
        gap_since_seen = now_hours - last_t
        if gap_since_seen > 14 * 24:
            continue  # gone too long to matter
        velocity = recent_velocity(obs, now_hours)
        shift_per_hour = MOMENTUM_BETA * velocity
        # Age: exact when created is known, else first seen + the prior.
        if uid in created:
            age_days = (t_end - created[uid]).total_seconds() / 86400.0
        else:
            first_t = obs[0][0]
            age_days = (now_hours - first_t) / 24.0 + AGE_PRIOR_DAYS
        mass = [0.0] * (places + 1)
        # Extra age hazard by band (top 10, 11-30, 31+).
        age_band = 0 if last_rank <= 10 else 1 if last_rank <= 30 else 2
        age_rate = AGE_EXTRA_HAZARD[age_band]
        age_survival = 1.0
        for h in range(1, window_hours + 1):
            # A game last seen a while ago has already travelled that long.
            eff_h = h + gap_since_seen
            # Momentum shifts the destination distribution: a game climbing
            # v places an hour is evaluated as if aiming shift places better.
            shift = shift_per_hour * h
            shift = max(-5.0, min(5.0, shift))
            dist = kernel(last_rank, eff_h)
            # Age factor: the kernel holds the average exit rate; old games
            # leave a little faster than average once past ~26 days.
            age_at_h = age_days + h / 24.0
            if age_at_h > AGE_ZONE_START:
                ramp = (age_at_h - AGE_ZONE_START) / (AGE_ZONE_FULL - AGE_ZONE_START)
                ramp = min(max(ramp, 0.0), 1.0)
                age_survival *= math.exp(-age_rate * ramp)
            for k, p_k in enumerate(
                    place_probs(dist, shift, places), start=1):
                if p_k > 0:
                    mass[k] += p_k * age_survival
        for k in range(1, places + 1):
            p = mass[k] / window_hours
            if OCC_BLEND > 0 and uid in occ:
                p = (1.0 - OCC_BLEND) * p + OCC_BLEND * occ[uid][k]
            # The kernel pools stable and volatile games, so its spread is
            # wider than any one game's; a sharpening exponent corrects
            # that over-dispersion (TEMP = 1 leaves the honest values).
            if TEMP != 1.0:
                p = p ** TEMP
            if p > 0.0005:
                prob.setdefault(uid, {})[k] = p

    # Per place, name the games whose probability beats their unnamed
    # value: the unnamed reserve divided by the games left unnamed.
    # The named mass is capped at 1 - UNNAMED_RESERVE.
    probabilities = []
    top_scores = {}
    for k in range(1, places + 1):
        cand = sorted(
            ((p[k], uid) for uid, p in prob.items() if k in p and p[k] > 0),
            reverse=True,
        )
        named = []
        named_mass = 0.0
        for p_val, uid in cand:
            if len(named) >= MAX_ROWS_PER_PLACE:
                break
            # A candidate too big for the headroom left is skipped, not
            # the end of the list: smaller candidates may still fit.
            if named_mass + p_val > 1.0 - UNNAMED_RESERVE:
                continue
            unnamed_count = max(close_size - len(named), 1)
            unnamed_value = UNNAMED_RESERVE / unnamed_count
            if p_val <= unnamed_value:
                break
            named.append((uid, p_val))
            named_mass += p_val
            top_scores[(uid, k)] = p_val
        for uid, p_val in named:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p_val, 6)})

    # Top 10: greedy maximum-weight assignment of games to places.
    pairs = sorted(
        ((p_val, uid, k) for (uid, k), p_val in top_scores.items()),
        reverse=True,
    )
    assigned_place = {}
    assigned_game = set()
    for p_val, uid, k in pairs:
        if k in assigned_place or uid in assigned_game:
            continue
        assigned_place[k] = uid
        assigned_game.add(uid)
        if len(assigned_place) == places:
            break
    top10 = [assigned_place[k] for k in range(1, places + 1) if k in assigned_place]
    if len(top10) < places:
        # Fill any gap from the last chart's order.
        for uid in sorted(last["games"], key=lambda u: last["games"][u]):
            if uid not in assigned_game:
                top10.append(uid)
                assigned_game.add(uid)
            if len(top10) == places:
                break
    return {"probabilities": probabilities, "top10": top10}


def main():
    snapshots, task, created = load_history(INPUT_DIR)
    forecast = build_forecast(snapshots, task, created)
    output_name = task.get("output", "forecast.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, output_name)
    with open(path, "w") as f:
        f.write(json.dumps(forecast) + "\n")
    return path


if __name__ == "__main__":
    print(main())
