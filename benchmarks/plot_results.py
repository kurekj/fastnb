"""benchmarks/plot_results.py
Generate publication-quality figures from benchmark_all_patients.json.

Usage:
    python benchmarks/plot_results.py

Outputs (DPI=300):
    benchmarks/results/fig2_speedup.png
    benchmarks/results/fig3_ram.png
    benchmarks/results/fig4_accuracy.png
    benchmarks/results/table1_timing.csv
    benchmarks/results/table2_accuracy.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_all_patients.json"
DPI = 300

COLORS = {
    "sklearn_silva": "#2166AC",
    "fastnb_t1": "#92C5DE",
    "fastnb_t4": "#4DAC26",
    "fastnb_t8": "#1A9641",
}


def load_data() -> dict:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def make_table1_timing(data: dict) -> pd.DataFrame:
    """Table 1: Per-patient timing comparison."""
    rows = []
    for pr in data["results"]:
        patient = pr["patient"]
        n_asvs = pr["n_sequences"]
        bm = {b["tool"]: b for b in pr["benchmarks"] if not b.get("skipped")}

        sk = bm.get("sklearn_silva", {})
        nb1 = bm.get("fastnb_t1", {})
        nb8 = bm.get("fastnb_t8", {})

        sk_time = sk.get("avg_time_s", 0)
        nb1_time = nb1.get("avg_time_s", 0)
        nb8_time = nb8.get("avg_time_s", 0)

        rows.append({
            "Patient": patient,
            "ASVs": n_asvs,
            "sklearn (s)": round(sk_time, 2),
            "fastnb 1T (s)": round(nb1_time, 2),
            "fastnb 8T (s)": round(nb8_time, 2),
            "Speedup (1T)": round(sk_time / nb1_time, 1) if nb1_time > 0 else 0,
            "Speedup (8T)": round(sk_time / nb8_time, 1) if nb8_time > 0 else 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table1_timing.csv", index=False)
    print(f"Saved: table1_timing.csv")
    return df


def make_table2_accuracy(data: dict) -> pd.DataFrame:
    """Table 2: Per-patient accuracy comparison."""
    rows = []
    for pr in data["results"]:
        acc = pr.get("accuracy", {})
        if acc.get("skipped"):
            continue
        rows.append({
            "Patient": pr["patient"],
            "Total ASVs": acc["total_asvs"],
            "Genus match": acc["genus_match"],
            "Match rate (%)": acc["genus_match_pct"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table2_accuracy.csv", index=False)
    print(f"Saved: table2_accuracy.csv")
    return df


def plot_speedup(data: dict) -> None:
    """Figure 2: Speedup bar chart across patients."""
    patients = []
    speedups_1t = []
    speedups_8t = []

    for pr in data["results"]:
        bm = {b["tool"]: b for b in pr["benchmarks"] if not b.get("skipped")}
        sk = bm.get("sklearn_silva", {})
        nb1 = bm.get("fastnb_t1", {})
        nb8 = bm.get("fastnb_t8", {})

        sk_time = sk.get("avg_time_s", 0)
        nb1_time = nb1.get("avg_time_s", 0)
        nb8_time = nb8.get("avg_time_s", 0)

        if nb1_time > 0 and sk_time > 0:
            patients.append(pr["patient"].replace("Patient.", "P"))
            speedups_1t.append(sk_time / nb1_time)
            speedups_8t.append(sk_time / nb8_time)

    x = np.arange(len(patients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, speedups_1t, width, label="fastnb (1 thread)",
           color=COLORS["fastnb_t1"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, speedups_8t, width, label="fastnb (8 threads)",
           color=COLORS["fastnb_t8"], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Speedup vs sklearn", fontsize=11)
    ax.set_xlabel("Patient sample", fontsize=11)
    ax.set_title("Classification speedup: fastnb vs sklearn MultinomialNB", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(patients)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_speedup.png", dpi=DPI)
    plt.close(fig)
    print(f"Saved: fig2_speedup.png")


def plot_ram(data: dict) -> None:
    """Figure 3: Process RSS comparison during classification."""
    sk_rams, nb_rams = [], []
    for pr in data["results"]:
        bm = {b["tool"]: b for b in pr["benchmarks"] if not b.get("skipped")}
        sk = bm.get("sklearn_silva", {})
        nb1 = bm.get("fastnb_t1", {})
        if sk.get("avg_peak_ram_mb", 0) > 0:
            sk_rams.append(sk["avg_peak_ram_mb"])
        if nb1.get("model_ram_mb", 0) > 0:
            nb_rams.append(nb1["model_ram_mb"])

    labels = ["sklearn\n(process RSS)", "fastnb\n(model only)"]
    values = [np.mean(sk_rams) if sk_rams else 0, np.mean(nb_rams) if nb_rams else 0]
    colors = [COLORS["sklearn_silva"], COLORS["fastnb_t1"]]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title("Memory usage: sklearn process RSS vs fastnb model", fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                f"{val:,.0f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_ram.png", dpi=DPI)
    plt.close(fig)
    print(f"Saved: fig3_ram.png")


def plot_accuracy(data: dict) -> None:
    """Figure 4: Genus-level agreement bar chart."""
    patients = []
    match_rates = []

    for pr in data["results"]:
        acc = pr.get("accuracy", {})
        if acc.get("skipped"):
            continue
        patients.append(pr["patient"].replace("Patient.", "P"))
        match_rates.append(acc["genus_match_pct"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(patients, match_rates, color="#4DAC26", edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, match_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Genus-level agreement (%)", fontsize=11)
    ax.set_xlabel("Patient sample", fontsize=11)
    ax.set_title("Classification agreement: fastnb vs sklearn (genus level)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_accuracy.png", dpi=DPI)
    plt.close(fig)
    print(f"Saved: fig4_accuracy.png")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print(f"Processing {len(data['results'])} patient results …\n")
    make_table1_timing(data)
    make_table2_accuracy(data)
    plot_speedup(data)
    plot_ram(data)
    plot_accuracy(data)
    print("\nDone — all figures and tables generated.")


if __name__ == "__main__":
    main()
