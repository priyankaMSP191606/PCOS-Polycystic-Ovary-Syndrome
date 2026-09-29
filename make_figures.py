"""Figures 2-5 and 7 for FedPCOS-XAI revision (v18), drawn from runs_v2 outputs.
Palette: Okabe-Ito (colour-blind safe); identity always also carried by labels or markers."""
import json, glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/home/claude/w2v2/runs_v2"; OUT = "/home/claude/figs_v18"; os.makedirs(OUT, exist_ok=True)
A = sorted(glob.glob(f"{R}/analysis_v1/*"))[-1]
M = ["mobilenetv2", "shufflenetv2", "alexnet", "densenet121", "vgg16", "googlenet", "resnet50", "secnn"]
NAME = dict(secnn="SE-CNN", alexnet="AlexNet", vgg16="VGG-16", resnet50="ResNet-50", densenet121="DenseNet-121",
            googlenet="GoogLeNet", mobilenetv2="MobileNetV2", shufflenetv2="ShuffleNetV2")
SEEDS = [42, 7, 99]; SC = {42: "#0072B2", 7: "#E69F00", 99: "#009E73"}; SM = {42: "o", 7: "s", 99: "^"}
INK, MUTED, GRID = "#1F2328", "#6B7280", "#E5E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK})

def save(fig, name):
    for ext, kw in (("png", {}), ("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

def rounds(exp, m, s):
    return [json.loads(l) for l in open(f"{R}/results/{exp}/{m}/seed{s}/rounds.jsonl")]
def result(exp, m, s):
    return json.load(open(f"{R}/results/{exp}/{m}/seed{s}/result.json"))

per = pd.read_csv(f"{A}/per_seed_results_v1.csv"); fed = per[per.exp == "fed"]

# ---------------- Figure 2: metrics per architecture (small multiples, one axis each) ----------------
cols = [("g_bal_acc", "Balanced accuracy"), ("g_sens", "Sensitivity"), ("g_spec", "Specificity"),
        ("worst_client_bal", "Worst-client balanced accuracy")]
fig, ax = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
order = fed.groupby("model").g_bal_acc.mean().sort_values().index.tolist()
y = np.arange(len(order))
for a, (c, t) in zip(ax, cols):
    for i, m in enumerate(order):
        v = fed[fed.model == m][c].values
        a.plot([v.min(), v.max()], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        for s in SEEDS:
            vv = fed[(fed.model == m) & (fed.seed == s)][c].values[0]
            a.scatter(vv, i, s=22, marker=SM[s], color=SC[s], edgecolor="white", lw=0.5, zorder=3)
        a.scatter(v.mean(), i, s=70, marker="|", color=INK, lw=1.6, zorder=4)
    a.set_title(t, loc="left", fontsize=9.5, fontweight="bold"); a.set_xlim(0.94, 1.002)
    a.xaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
ax[0].set_yticks(y); ax[0].set_yticklabels([NAME[m] for m in order])
h = [plt.Line2D([], [], marker=SM[s], color=SC[s], ls="", label=f"seed {s}") for s in SEEDS] + \
    [plt.Line2D([], [], marker="|", color=INK, ls="", ms=9, mew=1.6, label="mean")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "Figure2_metrics_v18")

# ---------------- Figure 3: convergence (mean client-validation BA per round) ----------------
fig, ax = plt.subplots(2, 4, figsize=(12, 5.4), sharex=True, sharey=True)
for a, m in zip(ax.flat, M):
    for s in SEEDS:
        rr = rounds("fed", m, s); x = [r["round"] for r in rr]; v = [r["val_mean_bal_acc"] for r in rr]
        a.plot(x, v, color=SC[s], lw=1.4, marker=SM[s], ms=2.8, label=f"seed {s}")
        sel = result("fed", m, s)["selected_round"]
        a.scatter(sel, v[sel - 1], s=90, marker="*", color=SC[s], edgecolor=INK, lw=0.6, zorder=5)
    a.axhline(0.5, color=MUTED, ls=(0, (2, 2)), lw=0.8)
    a.set_title(NAME[m], loc="left", fontsize=9.5, fontweight="bold"); a.set_ylim(0.45, 1.01)
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.set_xticks([1, 5, 10, 15, 20])
for a in ax[1]: a.set_xlabel("Communication round")
for a in ax[:, 0]: a.set_ylabel("Mean client-validation\nbalanced accuracy")
h = [plt.Line2D([], [], marker=SM[s], color=SC[s], label=f"seed {s}") for s in SEEDS] + \
    [plt.Line2D([], [], marker="*", color="white", markeredgecolor=INK, ms=10, ls="", label="selected round"),
     plt.Line2D([], [], color=MUTED, ls=(0, (2, 2)), label="chance (0.5)")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "Figure3_convergence_v18")

# ---------------- Figure 4: balanced accuracy and FRS (two panels, one axis each) ----------------
sm = pd.read_csv(f"{A}/summary_mean_sd_v1.csv"); sf = sm[sm.exp == "fed"].set_index("model")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
order = sf.g_bal_acc_mean.sort_values().index.tolist(); y = np.arange(len(order))
for a, (mu, sd, t, lim) in zip(ax, [("g_bal_acc_mean", "g_bal_acc_std", "A  Global-test balanced accuracy", (0.97, 1.002)),
                                     ("FRS_mean", "FRS_std", "B  Federated Reliability Score (exploratory)", (0.85, 1.0))]):
    for i, m in enumerate(order):
        a.errorbar(sf.loc[m, mu], i, xerr=sf.loc[m, sd], fmt="o", color="#0072B2", ms=6, capsize=3, lw=1.2)
        a.text(sf.loc[m, mu] + sf.loc[m, sd] + (lim[1] - lim[0]) * 0.015, i, f"{sf.loc[m, mu]:.3f}", va="center", fontsize=8, color=MUTED)
    a.set_title(t, loc="left", fontsize=9.5, fontweight="bold"); a.set_xlim(*lim); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    a.xaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
ax[0].set_yticks(y); ax[0].set_yticklabels([NAME[m] for m in order])
ax[1].text(0.0, -0.2, "Equal weights, τ = 0.10; stability from client-validation accuracy (rounds 16–20). Mean ± SD over three seeds.",
           transform=ax[1].transAxes, fontsize=7.5, color=MUTED)
fig.tight_layout(w_pad=3); save(fig, "Figure4_frs_v18")

# ---------------- Figure 5: failed vs corrected configurations ----------------
pairs = [("mobilenetv2", "fed_fp16", "fed", "FP16 (failed)", "FP32 (corrected)"),
         ("shufflenetv2", "fed_fp16", "fed", "FP16 (failed)", "FP32 (corrected)"),
         ("alexnet", "fed_lr1e3", "fed", "lr 1e-3 (failed)", "lr 1e-4 (corrected)"),
         ("vgg16", "fed_lr1e3", "fed", "lr 1e-3 (failed)", "lr 1e-4 (corrected)")]
fig, ax = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
for a, (m, bad, good, lb, lg) in zip(ax, pairs):
    for exp, col, lab in ((bad, "#D55E00", lb), (good, "#0072B2", lg)):
        for k, s in enumerate(SEEDS):
            rr = rounds(exp, m, s); x = [r["round"] for r in rr]; v = [r["val_mean_bal_acc"] for r in rr]
            v = [np.nan if (vv is None or (isinstance(vv, float) and np.isnan(vv))) else vv for vv in v]
            a.plot(x, v, color=col, lw=1.3, alpha=0.9, marker=SM[s], ms=2.5, label=lab if k == 0 else None)
    a.axhline(0.5, color=MUTED, ls=(0, (2, 2)), lw=0.8)
    a.set_title(NAME[m], loc="left", fontsize=9.5, fontweight="bold"); a.set_ylim(0.2, 1.02)
    a.set_xlabel("Communication round"); a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
ax[0].set_ylabel("Mean client-validation\nbalanced accuracy")
fig.tight_layout(); save(fig, "Figure5_failures_v18")

# ---------------- Figure 7: shortcut evidence ----------------
inv = pd.read_csv(f"{R}/file_inventory_v2.csv"); k = inv[inv.status == "kept"]
la = pd.read_csv(f"{R}/leak_audit_globaltest_vs_train_v2.csv")
sc = pd.read_csv(glob.glob(f"{R}/shortcutcnn_v2/*/shortcut_cnn_per_seed_v2.csv")[0])
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
a = ax[0]; p, n = k[k.label == 1], k[k.label == 0]
a.scatter(p.width, p.height, s=8, alpha=0.35, color="#E69F00", edgecolor="none", label=f"PCOS (n = {len(p):,})", rasterized=True)
a.scatter(n.width, n.height, s=26, alpha=0.9, color="#0072B2", marker="D", edgecolor="white", lw=0.4, label=f"non-PCOS (n = {len(n):,})")
a.set_xscale("log"); a.set_yscale("log"); a.legend(frameon=False, loc="upper left", fontsize=8)
a.text(0.97, 0.04, "Geometry-only random forest:\nbalanced accuracy 1.000", transform=a.transAxes, ha="right", va="bottom",
       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
a.set_xlabel("Image width (pixels, log scale)"); a.set_ylabel("Image height (pixels, log scale)")
a.set_title("A  Image dimensions by class", loc="left", fontweight="bold", fontsize=10)
a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
b = ax[1]; bins = np.arange(-0.5, la.nn_dist_train.max() + 1.5, 1)
b.axvspan(-0.5, 2.5, color="#FCE4D6", zorder=0)
b.hist([la[la.label == 0].nn_dist_train, la[la.label == 1].nn_dist_train], bins=bins, stacked=True,
       color=["#0072B2", "#E69F00"], label=["non-PCOS", "PCOS"], edgecolor="white", lw=0.5)
b.text(1.0, 95, "distance \u2264 2\n(duplicate after\nrotation or flip):\n0 images", fontsize=7.5, color="#A0400B", va="center", ha="center")
b.set_xlabel("Hamming distance to nearest training image\n(pHash, minimum over 8 rotations/flips)")
b.set_ylabel("Global-test images (n = 773)"); b.legend(frameon=False, fontsize=8, loc="upper right")
b.set_title("B  Transformation-aware leakage audit", loc="left", fontweight="bold", fontsize=10)
b.yaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)
c = ax[2]; inputs = ["original", "pad_square", "periphery_only", "border_removed", "centre_only"]
lab = {"original": "Original", "pad_square": "Pad to\nsquare", "periphery_only": "Periphery\nonly", "border_removed": "Border\nremoved", "centre_only": "Centre\nonly"}
mk = dict(zip(M, ["o", "s", "^", "D", "v", "P", "X", "*"])); pal = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000", "#999999"]
g = sc.groupby(["model", "input"]).bal_acc.mean()
for j, m in enumerate(M):
    xs = np.arange(len(inputs)) + (j - 3.5) * 0.07
    c.plot(xs, [g[(m, i)] for i in inputs], color=pal[j], lw=0.8, alpha=0.6)
    c.scatter(xs, [g[(m, i)] for i in inputs], marker=mk[m], color=pal[j], s=26, edgecolor="white", lw=0.4, label=NAME[m], zorder=3)
c.axhline(0.5, color=MUTED, ls=(0, (2, 2)), lw=0.8); c.text(4.35, 0.47, "chance", ha="right", fontsize=7.5, color=MUTED)
c.set_xticks(range(len(inputs))); c.set_xticklabels([lab[i] for i in inputs]); c.set_ylim(0.45, 1.02)
c.set_ylabel("Global-test balanced accuracy\n(mean of three seeds)")
c.set_title("C  Federated CNNs on manipulated inputs", loc="left", fontweight="bold", fontsize=10)
c.legend(frameon=False, fontsize=7, ncol=2, loc="lower left"); c.yaxis.grid(True, color=GRID, lw=0.6); c.set_axisbelow(True)
fig.tight_layout(w_pad=2.2); save(fig, "Figure7_shortcut_v18")
print("done")
