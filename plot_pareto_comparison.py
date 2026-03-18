"""
Pareto Front Comparison Plot: Top-left accuracy vs Bottom-right accuracy.
Reads wandb table JSONs or log files to extract accuracy data per method.
"""
import json
import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

BASE_LOG_DIR = r"d:\palora\logs\multimnist"


# ─── Data loading helpers ───────────────────────────────

def find_wandb_table(run_dir, split="test"):
    """Find the latest wandb table JSON."""
    patterns = [
        os.path.join(run_dir, "wandb", "run-*", "files", "media", "table", split, "*.table.json"),
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            best, best_epoch = None, -1
            for f in files:
                m = re.search(r'epoch=(\d+)', os.path.basename(f))
                if m:
                    ep = int(m.group(1))
                    if ep > best_epoch:
                        best_epoch = ep
                        best = f
            return best
    return None


def load_from_wandb_table(table_path):
    """Load (top_left_acc[], bottom_right_acc[]) from wandb table JSON."""
    with open(table_path, 'r') as f:
        data = json.load(f)
    cols = data["columns"]
    tl_idx = cols.index("acc/top-left")
    br_idx = cols.index("acc/bottom-right")
    tl = [row[tl_idx] for row in data["data"]]
    br = [row[br_idx] for row in data["data"]]
    return np.array(tl), np.array(br)


def load_last_epoch_from_log(log_path):
    """Parse log file to get the LAST validation epoch's per-ray accuracy data."""
    tl_all, br_all = [], []
    current_epoch_tl, current_epoch_br = [], []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines like: "0: {'avg_loss': ..., 'acc/top-left': 0.9434, 'acc/bottom-right': 0.9151, ...}"
            m = re.search(r"\d+:\s*\{.*?'acc/top-left':\s*([\d.]+).*?'acc/bottom-right':\s*([\d.]+)", line)
            if m:
                current_epoch_tl.append(float(m.group(1)))
                current_epoch_br.append(float(m.group(2)))
            
            # Epoch boundary: "Best results for VAL dataset"
            if "Best results for VAL dataset" in line:
                if current_epoch_tl:
                    tl_all = current_epoch_tl[:]
                    br_all = current_epoch_br[:]
                current_epoch_tl, current_epoch_br = [], []
    
    # In case last epoch didn't have "Best results" line yet
    if current_epoch_tl:
        tl_all = current_epoch_tl
        br_all = current_epoch_br
    
    return np.array(tl_all), np.array(br_all)


# ─── Method configs ─────────────────────────────────────

METHODS = [
    {
        "name": "PaLoRA",
        "dir": os.path.join(BASE_LOG_DIR, "palora", "2026-03-14", "16-36-30"),
        "color": "#4DB6AC",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "PaMaL",
        "dir": os.path.join(BASE_LOG_DIR, "pamal", "2026-03-14", "10-49-56"),
        "color": "#777777",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
    },
    {
        "name": "PaMaL-QD",
        "dir": os.path.join(BASE_LOG_DIR, "pamal_qd", "2026-03-17", "00-17-03"),
        "color": "#444444",
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 1.8,
    },
    {
        "name": "PaGeL [ours]",
        "dir": os.path.join(BASE_LOG_DIR, "pagel", "2026-03-17", "21-34-14"),
        "color": "#2196F3",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder_boost": 2,
    },
]


def load_last_population_results(run_dir):
    """Parse the last 'Population results' from the log file."""
    log_path = os.path.join(run_dir, "_multimnist.log")
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Population results:\s*(\{.*\})", line)
            if m:
                try:
                    last = eval(m.group(1))
                except:
                    pass
    return last


# ─── Main Plot ──────────────────────────────────────────

fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(9, 8),
    gridspec_kw={"height_ratios": [3, 1]})

table_rows = []

for cfg in METHODS:
    tl, br = None, None
    source = ""
    
    # Try wandb test table first
    table = find_wandb_table(cfg["dir"], "test")
    if table:
        tl, br = load_from_wandb_table(table)
        source = f"test (wandb)"
    else:
        # Try wandb val table
        table = find_wandb_table(cfg["dir"], "val")
        if table:
            tl, br = load_from_wandb_table(table)
            source = f"val (wandb)"
        else:
            # Parse log file
            log_path = os.path.join(cfg["dir"], "_multimnist.log")
            if os.path.exists(log_path):
                tl, br = load_last_epoch_from_log(log_path)
                source = f"log (last epoch)"
    
    if tl is None or len(tl) == 0:
        print(f"⚠ No data for {cfg['name']}")
        continue
    
    print(f"✓ {cfg['name']}: {len(tl)} points from {source}")
    
    # Sort by top-left accuracy for curve
    idx = np.argsort(tl)
    tl_s, br_s = tl[idx], br[idx]
    
    zb = cfg.get("zorder_boost", 0)
    
    # Plot curve
    ax.plot(tl_s, br_s,
        color=cfg["color"], linestyle=cfg["linestyle"],
        linewidth=cfg["linewidth"], label=cfg["name"],
        zorder=2 + zb,
    )
    # Plot points
    ax.scatter(tl, br,
        color=cfg["color"], marker=cfg["marker"],
        s=45, edgecolors="white", linewidths=0.6,
        zorder=3 + zb,
    )
    
    # Collect metrics
    pop = load_last_population_results(cfg["dir"])
    row = [cfg["name"]]
    if pop:
        row.extend([
            f"{pop.get('hypervolume', 0):.4f}",
            f"{pop.get('uniformity', 0):.2e}",
            f"{pop.get('spearman', 0):.4f}",
            str(pop.get('num_non_dominated', '-')),
        ])
    else:
        row.extend(["-", "-", "-", "-"])
    # Add avg accuracy
    row.append(f"{np.mean(tl):.4f}")
    row.append(f"{np.mean(br):.4f}")
    table_rows.append(row)

ax.set_xlabel("Top-left accuracy", fontsize=13)
ax.set_ylabel("Bottom-right accuracy", fontsize=13)
ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_title("MultiMNIST — Pareto Front Comparison", fontsize=14, fontweight='bold')

# ─── Metrics Table ──────────────────────────────────────
ax_table.axis("off")
col_labels = ["Method", "HV ↑", "Unif ↓", "Spearman", "#ND", "Avg TL", "Avg BR"]
tbl = ax_table.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.4)

# Style header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#333333")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(len(table_rows)):
    color = "#f0f0f0" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(color)

plt.tight_layout()

output_path = os.path.join(BASE_LOG_DIR, "pareto_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to {output_path}")
plt.close()

