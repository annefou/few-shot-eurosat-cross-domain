"""Headline replication figure: our results against the Guo et al. (2020) baselines.

Reads every results/*.json produced by scripts 01-03 and draws the comparison that
decides whether the replication succeeded:

  left  - accuracy vs. number of shots, published baseline as reference
  right - difference from the baseline in percentage points, with 95% CIs

Nothing is hard-coded: conditions, shot counts, accuracies, CIs and the cited work
all come from the result files. Run after the experiments:

    python 04_plot_replication.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
OUTPUT = FIGURES_DIR / "replication_comparison.png"

# colourblind-safe, ordered by training cost
CONDITION_COLORS = ["#0F7B6C", "#1F6FEB", "#B54708"]
BASELINE_COLOR = "#6B7280"


def condition_label(result):
    """Describe a run using only what the result file actually records."""
    if result.get("training_method") == "supervised classification":
        return f"Supervised, {result['n_train_epochs']} epochs"
    if "n_train_episodes" in result:
        return f"Episodic, {result['n_train_episodes']:,} episodes"
    return "Unlabelled condition"


def training_cost(result):
    """Sort key so conditions read from cheapest to most expensive training."""
    if result.get("training_method") == "supervised classification":
        return (0, result.get("n_train_epochs", 0))
    return (1, result.get("n_train_episodes", 0))


def shot_count(key):
    """'5way_20shot' -> 20."""
    match = re.search(r"(\d+)shot", key)
    if not match:
        raise ValueError(f"cannot read a shot count from {key!r}")
    return int(match.group(1))


def load_results():
    results = []
    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        with open(path) as handle:
            results.append(json.load(handle))
    if not results:
        raise SystemExit(
            f"No *_results.json in {RESULTS_DIR}/ - run the experiments first."
        )
    return sorted(results, key=training_cost)


def series(block):
    """{'5way_5shot': {...}} -> ([shots], [accuracy %], [ci95 %]) sorted by shots."""
    rows = sorted(((shot_count(k), v) for k, v in block.items()), key=lambda r: r[0])
    shots = [r[0] for r in rows]
    accuracy = [float(r[1]["accuracy"]) * 100 for r in rows]
    ci95 = [float(r[1]["ci95"]) * 100 for r in rows]
    return shots, accuracy, ci95


def main():
    results = load_results()
    reference = results[0]

    # every run must cite the same published numbers for one baseline band to be honest
    baselines = {json.dumps(r["guo_baselines"], sort_keys=True) for r in results}
    if len(baselines) > 1:
        raise SystemExit("Result files disagree on guo_baselines - cannot draw one baseline.")

    fig, (ax_acc, ax_diff) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # ---- left: absolute accuracy, published baseline as the reference band ----
    base_shots, base_acc, base_ci = series(reference["guo_baselines"])
    x = list(range(len(base_shots)))  # evenly spaced: 5, 20 and 50 are categories here
    ax_acc.fill_between(
        x,
        [a - c for a, c in zip(base_acc, base_ci)],
        [a + c for a, c in zip(base_acc, base_ci)],
        color=BASELINE_COLOR,
        alpha=0.18,
        linewidth=0,
    )
    ax_acc.plot(
        x,
        base_acc,
        color=BASELINE_COLOR,
        linestyle="--",
        marker="s",
        markersize=5,
        linewidth=1.8,
        label="Published baseline (95% CI)",
        zorder=3,
    )

    for result, color in zip(results, CONDITION_COLORS):
        shots, accuracy, ci95 = series(result["our_results"])
        ax_acc.errorbar(
            x,
            accuracy,
            yerr=ci95,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=condition_label(result),
            zorder=4,
        )

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([str(s) for s in base_shots])
    ax_acc.set_xlim(-0.18, len(x) - 0.82)
    ax_acc.set_xlabel("Labelled satellite images per class (shots)")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Cross-domain accuracy on EuroSAT", fontsize=12, pad=10)
    ax_acc.legend(frameon=False, fontsize=9, loc="lower right")

    # ---- right: agreement with the published result ----
    for result, color in zip(results, CONDITION_COLORS):
        shots, accuracy, ci95 = series(result["our_results"])
        _, baseline, _ = series(result["guo_baselines"])
        delta = [a - b for a, b in zip(accuracy, baseline)]
        ax_diff.errorbar(
            x,
            delta,
            yerr=ci95,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=condition_label(result),
        )

    ax_diff.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.8, zorder=1)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels([str(s) for s in base_shots])
    ax_diff.set_xlim(-0.18, len(x) - 0.82)
    ax_diff.set_xlabel("Labelled satellite images per class (shots)")
    ax_diff.set_ylabel("Difference from published (percentage points)")
    ax_diff.set_title("Agreement with the published result", fontsize=12, pad=10)
    ax_diff.annotate(
        "matches published",
        xy=(-0.12, 0),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=BASELINE_COLOR,
    )

    for ax in (ax_acc, ax_diff):
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#D1D5DB")
        ax.tick_params(colors="#374151", labelsize=9.5)

    method = reference["method"]
    fig.suptitle(
        f"{method}: {reference['training_domain'].split(' (')[0]} "
        f"→ {reference['test_domain'].split(' (')[0]}",
        fontsize=13.5,
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        f"Replication of {reference['replicates']} "
        f"(doi:{reference['replicates_doi']}). "
        "Error bars are 95% confidence intervals; random guessing on a 5-way task gives 20%.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )

    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")

    for result in results:
        shots, accuracy, ci95 = series(result["our_results"])
        _, baseline, _ = series(result["guo_baselines"])
        deltas = ", ".join(
            f"{s}-shot {a - b:+.1f}pp" for s, a, b in zip(shots, accuracy, baseline)
        )
        print(f"  {condition_label(result):<32} {deltas}")


if __name__ == "__main__":
    main()
