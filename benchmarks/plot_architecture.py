"""Generate architecture diagram (Fig 1) for SoftwareX article."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "SoftwareX_template" / "figures" / "fig1_architecture.png"

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis("off")

# Colors
C_INPUT = "#E3F2FD"
C_HASH  = "#BBDEFB"
C_GEMM  = "#1565C0"
C_SOFT  = "#42A5F5"
C_OUT   = "#E8F5E9"
C_MODEL = "#FFF3E0"
C_TEXT  = "white"

def box(x, y, w, h, label, color, fontsize=9, textcolor="black", bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor="#333", linewidth=1.2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor, weight=weight)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))

# Row 1: Pipeline flow
y1 = 3.0
box(0.1, y1, 1.6, 0.9, "DNA\nsequences\n(ASVs)", C_INPUT, 9)
arrow(1.7, y1+0.45, 2.1, y1+0.45)

box(2.1, y1, 2.0, 0.9, "MurmurHash3\n7-gram hashing\n+ L2 norm", C_HASH, 8)
arrow(4.1, y1+0.45, 4.5, y1+0.45)

box(4.5, y1, 2.2, 0.9, "Batch GEMM\n(float32, OpenMP)\nM @ F^T + prior", C_GEMM, 8, C_TEXT, True)
arrow(6.7, y1+0.45, 7.1, y1+0.45)

box(7.1, y1, 1.6, 0.9, "Softmax\n+ hierarchical\ntruncation", C_SOFT, 8, C_TEXT)
arrow(8.7, y1+0.45, 9.0, y1+0.45)

box(9.0, y1-0.05, 0.9, 1.0, "Taxon\n+\nConf.", C_OUT, 8)

# Row 2: Model input
y2 = 1.2
box(3.5, y2, 3.2, 0.9, "SILVA 138.2 trained model (float32)\n"
    "feature_log_prob: 79,664 × 8,192\n"
    "(2.45 GB)", C_MODEL, 8)
arrow(5.1, y2+0.9, 5.6, y1)

# Row 2: Feature matrix
box(0.3, y2, 2.4, 0.9, "Transposed features\n"
    "F^T ∈ R^{8192 × N}\n"
    "(fits in L2 cache)", C_MODEL, 8)
arrow(2.7, y2+0.9, 3.0, y1)

# Labels
ax.text(5.6, 0.3, "Reads model matrix once · OpenMP over 79,664 class rows · AVX2: 8 floats/cycle",
        ha="center", va="center", fontsize=8, style="italic", color="#555")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
