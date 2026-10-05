"""
2026 NCAA March Madness Final Four Championship Probability Model
=================================================================
Teams: Michigan, Arizona, UConn, Illinois
Metrics: Adjusted Offensive Efficiency, Adjusted Defensive Efficiency,
         eFG%, Veteran Guard Score, Rebounding Margin, Overall Adj. Efficiency Margin

Data sources: KenPom, ESPN, AP, CBS Sports reporting (as of March 31, 2026)
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# RAW STATS  (sourced from KenPom / tournament reporting)
# ─────────────────────────────────────────────────────────────────────────────
# adj_oe  : Adjusted Offensive Efficiency (points per 100 possessions)
# adj_de  : Adjusted Defensive Efficiency (points allowed per 100 possessions, lower = better)
# efg_pct : Effective Field Goal % (tournament, season-long where tourney unavailable)
# vet_grd : Veteran Guard Score 1–10  (experience, tournament reps, clutch play)
# reb_mrg : Rebounding margin (per game, regular season + tournament blended)
# adj_em  : Net Adjusted Efficiency Margin (KenPom, post-regionals)

TEAMS = {
    "Michigan": {
        "adj_oe":  121.8,   # KenPom top-2 all season; shooting 55.9% in tourney
        "adj_de":   82.8,   # leads country; +39.02 EM implies elite D
        "efg_pct":  0.591,  # 55.9% FG% in tourney; estimated eFG boost from 44.6% 3PT
        "vet_grd":  8.5,    # Roddy Gayle Jr. (senior), experienced backcourt
        "reb_mrg": +7.8,    # big front-line, dominant glass play
        "adj_em":  +39.02,  # #1 in country post-regionals (KenPom)
    },
    "Arizona": {
        "adj_oe":  120.4,   # 4th in adj. OE (KenPom); Big 12 POY Jaden Bradley
        "adj_de":   81.6,   # 3rd in adj. DE; 91st percentile man defense (Synergy)
        "efg_pct":  0.561,  # consistent 50%+ inside arc; 3rd in paint scoring
        "vet_grd":  9.0,    # Jaden Bradley (Big 12 POY, 2-yr starter); most experienced guard
        "reb_mrg": +5.4,    # deep roster; 60 paint points vs. Arkansas
        "adj_em":  +38.76,  # #2 in country post-regionals (KenPom)
    },
    "UConn": {
        "adj_oe":  114.2,   # 74th in adj. OE since March 1 but improving; 54% tournament poss.
        "adj_de":   88.9,   # excellent D; 91st-percentile man defense (Synergy)
        "efg_pct":  0.528,  # 52.2% of possessions scored last 3 games
        "vet_grd":  9.5,    # Karaban (3rd year under Hurley), dynasty experience; Mullins clutch
        "reb_mrg": +3.2,    # solid but not dominant; depth compensates
        "adj_em":  +28.4,   # top-10 KenPom; 3-time finalist pedigree
    },
    "Illinois": {
        "adj_oe":  131.2,   # #1 in KenPom adjusted OE (131.2 pts/100 poss.)
        "adj_de":   99.1,   # 28th in adj. DE regular season; .976 pts/poss in tourney
        "efg_pct":  0.590,  # 59% 2PT since March 1; dominant interior efficiency
        "vet_grd":  7.0,    # Kylan Boswell (sr guard, backcourt anchor); some youth at key spots
        "reb_mrg": +16.3,   # #3 in off. rebounding %; +16.3/game in NCAA tourney
        "adj_em":  +31.5,   # top-10 KenPom; offense is historic
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# METRIC WEIGHTS  (must sum to 1.0)
# Priority order set by user:
#   1. Adj. Offensive Efficiency  → 0.30
#   2. Rebounding Margin          → 0.22
#   3. Net Adj. Efficiency Margin → 0.18
#   4. Adj. Defensive Efficiency  → 0.13
#   5. Effective FG%              → 0.10
#   6. Veteran Guard Play         → 0.07
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "adj_oe":  0.30,  # #1 — scoring engine is the primary driver
    "reb_mrg": 0.22,  # #2 — second chances & defensive stops are huge in single-elim
    "adj_em":  0.18,  # #3 — holistic net efficiency, best overall predictor
    "adj_de":  0.13,  # #4 — defense matters but ranked below offense here
    "efg_pct": 0.10,  # #5 — shooting efficiency, partially captured by adj_oe
    "vet_grd": 0.07,  # #6 — experience helps but talent matters more
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize(values: dict, invert: bool = False) -> dict:
    """Min-max normalize to [0, 1]. Invert for metrics where lower = better."""
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo if hi != lo else 1e-9
    normed = {k: (v - lo) / span for k, v in values.items()}
    if invert:
        normed = {k: 1 - v for k, v in normed.items()}
    return normed


def softmax(scores: dict) -> dict:
    """Convert raw composite scores to probabilities via softmax."""
    exp_scores = {k: math.exp(v * 5) for k, v in scores.items()}  # scale=5 sharpens spread
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE NORMALIZED METRIC TABLES
# ─────────────────────────────────────────────────────────────────────────────

raw = {metric: {team: TEAMS[team][metric] for team in TEAMS} for metric in WEIGHTS}

normed = {
    "adj_oe":  normalize(raw["adj_oe"],  invert=False),
    "adj_de":  normalize(raw["adj_de"],  invert=True),   # lower DE is better
    "efg_pct": normalize(raw["efg_pct"], invert=False),
    "vet_grd": normalize(raw["vet_grd"], invert=False),
    "reb_mrg": normalize(raw["reb_mrg"], invert=False),
    "adj_em":  normalize(raw["adj_em"],  invert=False),
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE & WIN PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

composite = {}
for team in TEAMS:
    composite[team] = sum(WEIGHTS[m] * normed[m][team] for m in WEIGHTS)

win_prob = softmax(composite)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Michigan":  "\033[94m",   # blue
    "Arizona":   "\033[91m",   # red
    "UConn":     "\033[96m",   # cyan
    "Illinois":  "\033[93m",   # orange/yellow
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"

BAR_WIDTH = 40

def bar(prob: float, color: str) -> str:
    filled = round(prob * BAR_WIDTH)
    return color + "█" * filled + "░" * (BAR_WIDTH - filled) + RESET

def pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


print()
print(BOLD + "=" * 72 + RESET)
print(BOLD + "  🏀  2026 NCAA MARCH MADNESS — FINAL FOUR CHAMPIONSHIP MODEL" + RESET)
print(BOLD + "=" * 72 + RESET)
print(f"  Semifinal 1:  UConn  vs.  Illinois")
print(f"  Semifinal 2:  Michigan  vs.  Arizona")
print(f"  Championship: April 6 · Lucas Oil Stadium · Indianapolis")
print(BOLD + "=" * 72 + RESET)

# ── Overall Win Probability ───────────────────────────────────────────────────
print()
print(BOLD + "  CHAMPIONSHIP WIN PROBABILITY" + RESET)
print("  " + "─" * 68)

ranked = sorted(win_prob.items(), key=lambda x: x[1], reverse=True)
for rank, (team, prob) in enumerate(ranked, 1):
    col = COLORS[team]
    print(f"  {rank}. {col}{BOLD}{team:<12}{RESET}  {bar(prob, col)}  {GREEN}{pct(prob, 1)}{RESET}")

# ── Per-Metric Breakdown ──────────────────────────────────────────────────────
METRIC_LABELS = {
    "adj_oe":  ("Adj. Offensive Eff. (pts/100 poss)", "higher = better"),
    "adj_de":  ("Adj. Defensive Eff. (pts allowed/100)", "lower  = better"),
    "efg_pct": ("Effective FG%                    ", "higher = better"),
    "vet_grd": ("Veteran Guard Score (1–10)        ", "higher = better"),
    "reb_mrg": ("Rebounding Margin (per game)      ", "higher = better"),
    "adj_em":  ("Net Adj. Efficiency Margin (KenPom)", "higher = better"),
}

print()
print(BOLD + "  RAW STATS BY METRIC" + RESET)
print("  " + "─" * 68)
header = f"  {'Metric':<38}  {'Michigan':>9}  {'Arizona':>9}  {'UConn':>9}  {'Illinois':>9}"
print(BOLD + header + RESET)
print("  " + "─" * 68)

for m, (label, note) in METRIC_LABELS.items():
    row = f"  {label}  "
    vals = raw[m]
    best = max(vals.values()) if m != "adj_de" else min(vals.values())
    for team in ["Michigan", "Arizona", "UConn", "Illinois"]:
        v = vals[team]
        is_best = (v == best)
        fmt = f"{v:>9.3f}" if m == "efg_pct" else (f"{v:>9.1f}" if m != "vet_grd" else f"{v:>9.1f}")
        if is_best:
            row += GREEN + BOLD + fmt + RESET + "  "
        else:
            row += fmt + "  "
    print(row)
    print(f"    {BOLD}({note}){RESET}")

# ── Normalized Score Breakdown ────────────────────────────────────────────────
print()
print(BOLD + "  NORMALIZED SCORES (0–1 per metric, weight applied)" + RESET)
print("  " + "─" * 68)
print(BOLD + f"  {'Metric':<38}  {'Weight':>6}  {'Michigan':>9}  {'Arizona':>9}  {'UConn':>9}  {'Illinois':>9}" + RESET)
print("  " + "─" * 68)

for m, (label, _) in METRIC_LABELS.items():
    w = WEIGHTS[m]
    row = f"  {label}  {w:>5.0%}   "
    for team in ["Michigan", "Arizona", "UConn", "Illinois"]:
        score = normed[m][team] * w
        row += f"{score:>9.4f}  "
    print(row)

print("  " + "─" * 68)
total_row = f"  {'COMPOSITE SCORE':<38}  {'100%':>6}   "
for team in ["Michigan", "Arizona", "UConn", "Illinois"]:
    total_row += f"{composite[team]:>9.4f}  "
print(BOLD + total_row + RESET)

# ── Narrative Notes ───────────────────────────────────────────────────────────
print()
print(BOLD + "  ANALYST NOTES" + RESET)
print("  " + "─" * 68)
notes = {
    "Michigan":  (
        "📊 Best KenPom EM ever recorded (+39.02). Shooting 44.6% from 3 in tourney. "
        "Three projected 1st-round picks. Dominant interior defense."
    ),
    "Arizona":   (
        "🐻 Jaden Bradley is the most seasoned guard in the field (Big 12 POY). "
        "Best defensive efficiency; 91st-pctile man D. Deep, balanced scoring."
    ),
    "UConn":     (
        "🏆 Highest veteran guard score — Karaban is a 2x champion. Hurley dynasty "
        "experience is unmatched. Won despite 19-pt halftime deficit vs. Duke."
    ),
    "Illinois":  (
        "🔥 #1 adjusted offense (131.2 pts/100). Tallest team in D-I. "
        "+16.3 rebounding margin in tourney. Defense elevated to elite level in March."
    ),
}
for team, note in notes.items():
    col = COLORS[team]
    prob = win_prob[team]
    print(f"  {col}{BOLD}{team}{RESET} ({pct(prob)}): {note}")
    print()

print(BOLD + "=" * 72 + RESET)
print("  Model: Weighted composite → softmax probability | Data: KenPom / ESPN")
print(BOLD + "=" * 72 + RESET)
print()
