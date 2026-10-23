"""Matplotlib figures for 01_icu_mortality.py (not traced by LecTrace; the lecture passes the figures to plot()).

Same colours as the report: Okabe-Ito blue for `no_llm`, vermillion for Llemr.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_SMALL = "#0072B2"  # no_llm
C_LLEMR = "#D55E00"  # Llemr-v1
GREY = "#6b6b6b"

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path",  # text as outlines: looks the same in every browser, no font needed
})


def lactate_figure(small: dict[int, float], llemr: dict[int, float]):
    """Change of the mean predicted risk (percentage points) relative to a latest lactate of 1 mmol/L."""
    targets = sorted(small)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    for values, colour, label in ((small, C_SMALL, "no_llm (0.29M parameters)"), (llemr, C_LLEMR, "Llemr-v1, fine-tuned (7.1B)")):
        change = [100 * (values[t] - values[targets[0]]) for t in targets]
        ax.plot(targets, change, marker="o", color=colour, lw=2, label=label)
        ax.annotate(f"{change[-1]:+.2f}", (targets[-1], change[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=colour, fontweight="bold")
    ax.set_xticks(targets)
    ax.set_xlim(targets[0] - 0.3, targets[-1] + 1.0)
    ax.set_xlabel("latest lactate of the patient set to (mmol/L)")
    ax.set_ylabel("change of mean risk vs. 1 mmol/L\n(percentage points)")
    ax.set_title("633 test patients, one lab value edited (recorded runs)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def shuffle_figure(small: dict[str, float], llemr: dict[str, float]):
    """Test AUROC with the real measured values and with the values shuffled between patients."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.34
    for k, (name, values, colour) in enumerate((("no_llm", small, C_SMALL), ("Llemr-v1", llemr, C_LLEMR))):
        for j, (key, alpha, label) in enumerate((("original", 1.0, "real values"), ("shuffled", 0.35, "values shuffled between patients"))):
            x = k + (j - 0.5) * width
            ax.bar(x, values[key], width * 0.92, color=colour, alpha=alpha, edgecolor=colour, lw=1.2)
            ax.text(x, values[key] + 0.008, f"{values[key]:.3f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no_llm\n(0.29M parameters)", "Llemr-v1, fine-tuned\n(7.1B parameters)"])
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("test AUROC (axis starts at chance, 0.5)")
    ax.set_title("What the measured values are worth")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=a) for a in (1.0, 0.35)],
              labels=["real values (dark)", "values shuffled between patients (light)"], frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig
