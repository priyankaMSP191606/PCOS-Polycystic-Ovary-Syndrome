#!/usr/bin/env python3
"""
fedpcos_pipeline.py - FedPCOS-XAI analysis pipeline (single file, GPU).

Stages (each resumable; a finished run is never retrained or overwritten):
  regroup     : transformation-aware duplicate groups - pHash under the 8 dihedral transforms,
                Hamming <= 2 under any transform merged by union-find; label-conflict check
  split       : group-safe split - 20% global test; Dirichlet(alpha) client assignment over K clients;
                per-client train / validation / test
  leakaudit   : rotation-invariant nearest-neighbour distances for every pair of partitions
  shortcut    : random-forest probes on image geometry and file metadata
  fed         : FedAvg (all batch-normalization parameters and buffers averaged); round selected on
                mean client-validation balanced accuracy
  central     : pooled-data baseline, same epoch budget
  local       : local-only baseline (each client trains alone)
  analyze     : per-seed tables, mean +- SD, paired group bootstrap (95% and Bonferroni-adjusted CIs),
                FRS and its sensitivity, confusion counts, communication and compute cost
  xai         : Grad-CAM deletion faithfulness vs a spatially matched random control,
                weight-randomization sanity test, cross-seed stability
  shortcutcnn : trained models scored under input manipulations (frame removed, centre-only,
                periphery-only, aspect-preserving padding)
  all         : every stage in order

Every round of every run is written to a JSONL log; every call's arguments are saved under <out>/args/.

Example:
  python fedpcos_pipeline.py --stage all --inventory file_inventory.csv \
      --data_root /path/to/PCOS_figshare_ds --out runs_v2
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, time, random, copy, platform, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as _tmp
_tmp.set_sharing_strategy("file_system")         # v6.3: avoid 'Too many open files' in long queues
try:
    import resource as _res
    _soft, _hard = _res.getrlimit(_res.RLIMIT_NOFILE)
    _res.setrlimit(_res.RLIMIT_NOFILE, (min(max(_soft, 65535), _hard), _hard))
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats

ALL_MODELS = ["secnn", "alexnet", "vgg16", "resnet50", "densenet121",
              "googlenet", "mobilenetv2", "shufflenetv2"]


# --------------------------------------------------------------------------- utils
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


STAMP = time.strftime("%Y%m%d_%H%M%S")


def jdump(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f: json.dump(o, f, indent=2, default=float)


def jl(p, o):
    with open(p, "a") as f: f.write(json.dumps(o, default=float) + "\n")


def args():
    a = argparse.ArgumentParser()
    a.add_argument("--stage", default="all",
                   choices=["all", "regroup", "split", "leakaudit", "shortcut", "fed", "central", "local", "analyze", "xai", "shortcutcnn"])
    a.add_argument("--inventory", default="/workspace/file_inventory.csv", help="ORIGINAL (v1) inventory")
    a.add_argument("--hamming", type=int, default=2, help="near-duplicate threshold (bits)")
    a.add_argument("--data_root", default="/workspace/data/PCOS_figshare_ds",
                   help="dataset folder containing PCOS/infected (or its parent containing data/PCOS)")
    a.add_argument("--out", default="runs_v2")
    a.add_argument("--models", default=",".join(ALL_MODELS))
    a.add_argument("--seeds", default="42,7,99")
    a.add_argument("--split_seed", type=int, default=42)
    a.add_argument("--clients", type=int, default=5)
    a.add_argument("--alpha", type=float, default=2.0)
    a.add_argument("--test_frac", type=float, default=0.20)
    a.add_argument("--rounds", type=int, default=20)
    a.add_argument("--local_epochs", type=int, default=2)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--bs", type=int, default=32)
    a.add_argument("--img", type=int, default=224)
    a.add_argument("--vflip", type=int, default=0, help="1 = add vertical flip (ablation)")
    a.add_argument("--tag", default="", help="suffix for experiment name, e.g. _vflip")
    a.add_argument("--pretrained", type=int, default=1)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--amp", type=int, default=1)
    a.add_argument("--save_model", type=int, default=1)
    a.add_argument("--n_boot", type=int, default=2000)
    a.add_argument("--xai_per_class", type=int, default=30)
    a.add_argument("--xai_max_errors", type=int, default=40)
    return a.parse_args()


# --------------------------------------------------------------------------- split
def strat_group_split(df, frac, seed):
    n = max(2, round(1 / frac))
    try:
        tr, te = next(StratifiedGroupKFold(n, shuffle=True, random_state=seed)
                      .split(df, df.label, df.group_id))
    except ValueError:  # too few groups of one class -> fall back to random group split
        g = df.group_id.unique(); rng = np.random.default_rng(seed); rng.shuffle(g)
        teg = set(g[:max(1, int(len(g) * frac))])
        m = df.group_id.isin(teg).values; tr, te = np.where(~m)[0], np.where(m)[0]
    return df.iloc[tr].copy(), df.iloc[te].copy()


def dirichlet_groups(dev, K, alpha, seed, min_groups=10):
    """Dirichlet label-skew allocation at GROUP level (a group never crosses clients)."""
    g = dev.groupby("group_id").label.first().reset_index()
    rng = np.random.default_rng(seed)
    for _ in range(200):
        assign = {}
        for c in sorted(g.label.unique()):
            ids = g[g.label == c].group_id.values.copy(); rng.shuffle(ids)
            p = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                for i in part: assign[i] = k
        cnt = pd.Series(assign).value_counts()
        ok = len(cnt) == K and cnt.min() >= min_groups
        # every client must hold both classes
        lab = g.set_index("group_id").label
        both = all(lab[[i for i, k in assign.items() if k == kk]].nunique() == 2 for kk in range(K))
        if ok and both: return assign
    raise RuntimeError("Could not draw a valid Dirichlet partition; lower alpha constraints.")


# --------------------------------------------------------------------------- v6: regroup
def _dihedral(im):
    out = []
    for f in (im, im.transpose(Image.FLIP_LEFT_RIGHT)):
        for k in range(4): out.append(f.rotate(90 * k, expand=True))
    return out                                    # out[0] = original orientation


def _phash64(im):
    import imagehash
    return int(str(imagehash.phash(im)), 16)


def _popcnt(x):
    x = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(x, axis=-1).sum(-1).astype(np.int64)


def _hash_table(paths, root):
    """64-bit pHash of every image under the 8 dihedral transforms + size/file metadata."""
    H = np.zeros((len(paths), 8), np.uint64); meta = []
    for i, p in enumerate(paths):
        f = img_path(root, p)
        with Image.open(f) as im:
            w, h = im.size; fmt = (im.format or Path(f).suffix.lstrip(".")).lower()
            g = im.convert("L")
            H[i] = [_phash64(t) for t in _dihedral(g)]
        meta.append(dict(width=w, height=h, file_bytes=os.path.getsize(f), fmt=fmt))
        if i % 500 == 0: print(f"  hashed {i}/{len(paths)}", flush=True)
    return H, pd.DataFrame(meta)


def _min_dihedral_dist(Hq, Href):
    """For every query row: min Hamming distance over 8 transforms to any reference image (orientation 0)
    and the index of that reference image."""
    ref = Href[:, 0]; best = np.full(len(Hq), 99); arg = np.full(len(Hq), -1)
    for i in range(len(Hq)):
        d = np.min([_popcnt(np.bitwise_xor(ref, Hq[i, g])) for g in range(8)], axis=0)
        j = int(np.argmin(d)); best[i] = int(d[j]); arg[i] = j
    return best, arg


def stage_regroup(a):
    """Transformation-aware grouping on the SHA-256-unique images of the original inventory."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dst = out / "file_inventory_v2.csv"
    if dst.exists(): print(f"[keep] {dst} exists"); return
    inv = pd.read_csv(a.inventory)
    uniq_mask = inv.status.isin(["kept", "excluded_label_conflict"])
    U = inv[uniq_mask].reset_index().rename(columns={"index": "inv_row"})
    print(f"[regroup] {len(U)} SHA-256-unique images")
    H, meta = _hash_table(U.path.tolist(), a.data_root)
    parent = list(range(len(U)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ref = H[:, 0]; n_edges = 0; n_rot = 0
    for i in range(len(U)):
        d_all = np.stack([_popcnt(np.bitwise_xor(ref, H[i, g])) for g in range(8)])   # 8 x N
        d = d_all.min(0); d[i] = 99
        for j in np.where(d <= a.hamming)[0]:
            n_edges += 1; n_rot += int(d_all[0, j] > a.hamming)
            ri, rj = find(i), find(int(j))
            if ri != rj: parent[ri] = rj
    root = np.array([find(i) for i in range(len(U))])
    U["group_v2"] = pd.factorize(root)[0]
    conf = U.groupby("group_v2").label.nunique(); bad = set(conf[conf > 1].index)
    U["status_v2"] = np.where(U.group_v2.isin(bad), "excluded_label_conflict", "kept")
    U = pd.concat([U, meta], axis=1)
    inv2 = inv.copy(); inv2["status_v1"] = inv2.status
    inv2.loc[U.inv_row, "status"] = U.status_v2.values
    inv2.loc[U.inv_row, "group_id"] = U.group_v2.values
    for c in ("width", "height", "file_bytes", "fmt"): inv2.loc[U.inv_row, c] = U[c].values
    inv2.to_csv(dst, index=False)
    np.save(out / "phash_dihedral_v2.npy", H); U[["path"]].to_csv(out / "phash_dihedral_v2_index.csv", index=False)
    k = U[U.status_v2 == "kept"]
    audit = dict(unique_images=len(U), edges_le_threshold=n_edges, edges_only_after_transform=n_rot,
                 groups_v2=int(U.group_v2.nunique()), conflict_groups_v2=len(bad),
                 conflict_images_v2=int(U.group_v2.isin(bad).sum()), kept_v2=len(k),
                 kept_pcos=int(k.label.sum()), kept_healthy=int((k.label == 0).sum()),
                 kept_groups_v2=int(k.group_v2.nunique()), hamming=a.hamming,
                 groups_v1_kept=int(inv[inv.status == "kept"].group_id.nunique()))
    jdump(audit, out / "regroup_audit_v2.json"); print(json.dumps(audit, indent=1))


def stage_leakaudit(a):
    """Rotation-invariant nearest-neighbour distance for every image, reported per partition pair."""
    out = Path(a.out); m = pd.read_csv(out / "split_manifest_v1.csv")
    H = np.load(out / "phash_dihedral_v2.npy"); idx = pd.read_csv(out / "phash_dihedral_v2_index.csv")
    pos = {p: i for i, p in enumerate(idx.path)}; Hm = H[[pos[p] for p in m.path]]
    m["role"] = m.partition.str.replace(r"client\d+_", "", regex=True).replace({"global_test": "global_test"})
    rows, per_img = [], []
    for q in ("global_test", "test", "val", "train"):
        for r in ("train", "val", "test", "global_test"):
            qi = np.where(m.role == q)[0]; ri = np.where(m.role == r)[0]
            if not len(qi) or not len(ri): continue
            d = np.full(len(qi), 99)
            for n, i in enumerate(qi):
                cand = ri[m.group_id.values[ri] != m.group_id.values[i]]
                if len(cand): d[n] = np.min([_popcnt(np.bitwise_xor(Hm[cand, 0], Hm[i, g])).min() for g in range(8)])
            rows.append(dict(query=q, reference=r, n_query=len(qi), le2=int((d <= 2).sum()),
                             d3_6=int(((d > 2) & (d <= 6)).sum()), gt6=int((d > 6).sum())))
            if q == "global_test" and r == "train":
                per_img = pd.DataFrame(dict(path=m.path.values[qi], label=m.label.values[qi], nn_dist_train=d))
    t = pd.DataFrame(rows); t.to_csv(out / "leak_audit_by_partition_v2.csv", index=False)
    if len(per_img): per_img.to_csv(out / "leak_audit_globaltest_vs_train_v2.csv", index=False)
    print(t.to_string(index=False))


def stage_shortcut(a):
    """Metadata-only probes (no pixels): geometry only, and geometry + file metadata."""
    from sklearn.ensemble import RandomForestClassifier
    out = Path(a.out); m = pd.read_csv(out / "split_manifest_v1.csv")
    inv = pd.read_csv(out / "file_inventory_v2.csv")[["path", "width", "height", "file_bytes", "fmt"]]
    d = m.merge(inv, on="path", how="left")
    d["aspect"] = d.width / d.height; d["pixels"] = d.width * d.height
    d["fmt_code"] = pd.factorize(d.fmt)[0]
    tr, te = d[d.partition != "global_test"], d[d.partition == "global_test"]
    res = []
    for name, cols in (("geometry_only", ["width", "height", "aspect", "pixels"]),
                       ("geometry_plus_file", ["width", "height", "aspect", "pixels", "file_bytes", "fmt_code"])):
        for seed in (42, 7, 99):
            rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(tr[cols], tr.label)
            p = rf.predict_proba(te[cols])[:, 1]; mt = metrics(te.label.values, p)
            res.append(dict(features=name, seed=seed, bal_acc=mt["bal_acc"], auc=mt["auc"],
                            tn=mt["tn"], fp=mt["fp"], fn=mt["fn"], tp=mt["tp"]))
    r = pd.DataFrame(res); r.to_csv(out / "shortcut_probe_v2.csv", index=False)
    size_tab = d.groupby("label").apply(lambda g: pd.Series(dict(
        n=len(g), distinct_sizes=g[["width", "height"]].drop_duplicates().shape[0],
        median_w=g.width.median(), median_h=g.height.median()))).reset_index()
    size_tab.to_csv(out / "image_dimensions_by_class_v2.csv", index=False)
    print(r.groupby("features")[["bal_acc", "auc"]].agg(["mean", "std"]).to_string()); print(size_tab.to_string(index=False))


def stage_split(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "split_manifest_v1.csv").exists():
        import hashlib
        h = hashlib.sha256((out / "split_manifest_v1.csv").read_bytes()).hexdigest()
        print(f"[keep] existing split reused, not overwritten (sha256 {h[:12]})"); return
    inv = pd.read_csv(out / "file_inventory_v2.csv")      # v6: transformation-aware groups
    kept = inv[inv.status == "kept"].copy()
    kept["label"] = kept.label.astype(int)
    dev, gtest = strat_group_split(kept, a.test_frac, a.split_seed)
    gtest["partition"] = "global_test"; gtest["client"] = -1
    assign = dirichlet_groups(dev, a.clients, a.alpha, a.split_seed)
    dev["client"] = dev.group_id.map(assign)
    parts = [gtest]
    for k in range(a.clients):
        c = dev[dev.client == k]
        trval, te = strat_group_split(c, 0.15, a.split_seed + k)
        tr, va = strat_group_split(trval, 0.15 / 0.85, a.split_seed + 100 + k)
        for d, name in ((tr, "train"), (va, "val"), (te, "test")):
            d = d.copy(); d["partition"] = f"client{k}_{name}"; parts.append(d)
    man = pd.concat(parts)[["path", "label", "group_id", "sha256", "client", "partition"]]
    assert man.path.is_unique and len(man) == len(kept)
    assert (man.groupby("group_id").partition.nunique() == 1).all(), "group leakage"
    man.to_csv(out / "split_manifest_v1.csv", index=False)
    import hashlib
    jdump(dict(sha256=hashlib.sha256((out / "split_manifest_v1.csv").read_bytes()).hexdigest(),
               created=STAMP), out / "split_manifest_v1.sha256.json")
    # composition table: unique and post-oversampling counts
    rows = []
    for p, d in man.groupby("partition"):
        n0, n1 = int((d.label == 0).sum()), int((d.label == 1).sum())
        r = dict(partition=p, n=len(d), healthy=n0, pcos=n1, pcos_prev=n1 / len(d), groups=d.group_id.nunique())
        if p.endswith("_train"):
            r["n_after_oversample"] = 2 * max(n0, n1)
        rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "client_composition_v1.csv", index=False)
    audit = dict(inventory_total=len(inv), status_counts=inv.status.value_counts().to_dict(),
                 kept=len(kept), kept_pcos=int(kept.label.sum()),
                 kept_healthy=int((kept.label == 0).sum()), groups=int(kept.group_id.nunique()),
                 split_seed=a.split_seed, alpha=a.alpha, clients=a.clients, test_frac=a.test_frac)
    jdump(audit, out / "split_audit_v1.json")
    print(comp.to_string(index=False)); print(json.dumps(audit, indent=1))


# --------------------------------------------------------------------------- data
def img_path(root, p):
    """Map an inventory path like data/PCOS/infected/x.jpg to a file on disk."""
    root = Path(root)
    if (root / "data" / "PCOS").is_dir(): return str(root / p)
    return str(root / (p[5:] if p.startswith("data/") else p))


def preflight(a):
    """Stop early with a clear message if images cannot be found."""
    m = pd.read_csv(Path(a.out) / "split_manifest_v1.csv") if (Path(a.out) / "split_manifest_v1.csv").exists() \
        else pd.read_csv(a.inventory).query("status in ['kept', 'excluded_label_conflict']")
    miss = [p for p in m.path if not os.path.exists(img_path(a.data_root, p))]
    if miss:
        raise SystemExit(f"ERROR: {len(miss)} of {len(m)} images not found under --data_root {a.data_root}\n"
                         f"  first missing: {img_path(a.data_root, miss[0])}\n"
                         f"  --data_root must be the folder that contains PCOS/infected and PCOS/noninfected")
    print(f"[ok] all {len(m)} images found under {a.data_root}")



MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tfm(a, train):
    if train:
        t = [T.Resize((a.img, a.img)), T.RandomHorizontalFlip(),
             T.RandomRotation(10), T.ColorJitter(0.1, 0.1)]
        if a.vflip: t.append(T.RandomVerticalFlip())
    else:
        t = [T.Resize((a.img, a.img))]
    return T.Compose(t + [T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(s, df, root, tf):
        s.p = [img_path(root, p) for p in df.path]; s.y = df.label.astype(int).tolist(); s.tf = tf
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        with Image.open(s.p[i]) as im: x = s.tf(im.convert("RGB"))
        return x, s.y[i]


def loader(df, a, train, seed=0):
    ds = DS(df, a.data_root, tfm(a, train))
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(batch_size=a.bs, num_workers=a.workers, pin_memory=True,
              persistent_workers=False)          # v6.3: workers closed after each pass (avoids fd exhaustion)
    if train:  # class-balanced oversampling of the TRAIN partition only
        y = np.array(ds.y); cnt = np.bincount(y, minlength=2)
        w = (1.0 / cnt[y]).astype(np.float64)
        smp = WeightedRandomSampler(torch.tensor(w), num_samples=int(2 * cnt.max()),
                                    replacement=True, generator=g)
        return DataLoader(ds, sampler=smp, worker_init_fn=lambda i: np.random.seed(seed + i), **kw)
    return DataLoader(ds, shuffle=False, **kw)


def load_manifest(a):
    m = pd.read_csv(Path(a.out) / "split_manifest_v1.csv")
    K = int(m.client.max()) + 1
    parts = [{s: m[m.partition == f"client{k}_{s}"] for s in ("train", "val", "test")} for k in range(K)]
    return m, parts, m[m.partition == "global_test"]


# --------------------------------------------------------------------------- models
class SE(nn.Module):
    def __init__(s, c, r):
        super().__init__()
        s.fc = nn.Sequential(nn.Linear(c, r), nn.ReLU(), nn.Linear(r, c), nn.Sigmoid())
    def forward(s, x):
        w = s.fc(x.mean((2, 3)))
        return x * w[:, :, None, None]


class SECNN(nn.Module):
    """Squeeze-and-Excitation CNN (1.22 M params; layout matches the earlier secnn_s42.pt)."""
    def __init__(s, C=2):
        super().__init__()
        L, cin = [], 3
        for i, (c, r) in enumerate([(32, 4), (64, 4), (128, 8), (256, 16)]):
            L += [nn.Conv2d(cin, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                  nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), SE(c, r),
                  nn.MaxPool2d(2) if i < 3 else nn.AdaptiveAvgPool2d(1)]
            cin = c
        s.features = nn.Sequential(*L)
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, C))
    def forward(s, x): return s.head(s.features(x))


def build(name, pretrained, C=2):
    tv = torchvision.models
    W = lambda enum: enum.DEFAULT if pretrained else None
    if name == "secnn": m = SECNN(C)
    elif name == "alexnet":
        m = tv.alexnet(weights=W(tv.AlexNet_Weights)); m.classifier[6] = nn.Linear(4096, C)
    elif name == "vgg16":
        m = tv.vgg16(weights=W(tv.VGG16_Weights)); m.classifier[6] = nn.Linear(4096, C)
    elif name == "resnet50":
        m = tv.resnet50(weights=W(tv.ResNet50_Weights)); m.fc = nn.Linear(2048, C)
    elif name == "densenet121":
        m = tv.densenet121(weights=W(tv.DenseNet121_Weights)); m.classifier = nn.Linear(1024, C)
    elif name == "googlenet":
        if pretrained:
            m = tv.googlenet(weights=tv.GoogLeNet_Weights.DEFAULT)  # torchvision drops aux heads itself
        else:
            m = tv.googlenet(weights=None, aux_logits=False, init_weights=True, transform_input=True)  # same forward as pretrained
        m.fc = nn.Linear(1024, C)
    elif name == "mobilenetv2":
        m = tv.mobilenet_v2(weights=W(tv.MobileNet_V2_Weights)); m.classifier[1] = nn.Linear(1280, C)
    elif name == "shufflenetv2":
        m = tv.shufflenet_v2_x1_0(weights=W(tv.ShuffleNet_V2_X1_0_Weights)); m.fc = nn.Linear(1024, C)
    else: raise ValueError(name)
    return m


def cam_layer(m, name):
    return {"secnn": lambda: m.features[30], "alexnet": lambda: m.features[11],
            "vgg16": lambda: m.features[29], "resnet50": lambda: m.layer4,
            "densenet121": lambda: m.features.denseblock4, "googlenet": lambda: m.inception5b,
            "mobilenetv2": lambda: m.features[18], "shufflenetv2": lambda: m.conv5}[name]()


CAM_LAYER_NAMES = {"secnn": "features.30 (last SE block)", "alexnet": "features.11 (ReLU after conv5)",
                   "vgg16": "features.29 (ReLU after conv5_3)", "resnet50": "layer4",
                   "densenet121": "features.denseblock4", "googlenet": "inception5b",
                   "mobilenetv2": "features.18", "shufflenetv2": "conv5"}


def n_params(m): return sum(p.numel() for p in m.parameters())


def state_bytes(m): return sum(v.numel() * v.element_size() for v in m.state_dict().values())


# --------------------------------------------------------------------------- train / eval
class Focal(nn.Module):
    def __init__(s, g=2.0): super().__init__(); s.g = g
    def forward(s, logit, y):
        ce = F.cross_entropy(logit, y, reduction="none"); p = torch.exp(-ce)
        return ((1 - p) ** s.g * ce).mean()


def train_epochs(model, dl, a, dev, epochs, opt):
    model.train(); crit = Focal(); scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp) and dev.type == "cuda")
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=bool(a.amp) and dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()


@torch.no_grad()
def predict(model, dl, dev, amp=True):
    model.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast(dev.type, enabled=bool(amp) and dev.type == "cuda"):
            P.append(F.softmax(model(x.to(dev)).float(), 1)[:, 1].cpu()); Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


def metrics(y, p1, thr=0.5):
    """Positive class = PCOS (label 1). Decision threshold 0.5 on P(PCOS). Zero-division -> 0."""
    yh = (p1 >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    d = lambda a_, b_: a_ / b_ if b_ else 0.0
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    prec, npv = d(tp, tp + fp), d(tn, tn + fn)
    f1 = d(2 * prec * sens, prec + sens)
    f1_neg = d(2 * npv * spec, npv + spec)
    try: auc = roc_auc_score(y, p1) if len(set(y)) == 2 else float("nan")
    except ValueError: auc = float("nan")
    return dict(acc=d(tp + tn, len(y)), bal_acc=(sens + spec) / 2, sens=sens, spec=spec, prec=prec,
                npv=npv, f1=f1, macro_f1=(f1 + f1_neg) / 2, auc=auc,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))


def fedavg(sds, w):
    """Weighted average of ALL state entries (weights, BN affine params AND BN running stats).
    Integer buffers (num_batches_tracked) are taken from client 0."""
    out = {}
    for k in sds[0]:
        if not sds[0][k].is_floating_point(): out[k] = sds[0][k].clone(); continue
        out[k] = sum(wi * sd[k].float() for wi, sd in zip(w, sds)).to(sds[0][k].dtype)
    return out


def cpu_sd(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def exp_name(kind, a): return f"{kind}{a.tag}"


def run_dir(a, kind, model, seed): return Path(a.out) / "results" / exp_name(kind, a) / model / f"seed{seed}"


def env_info(dev):
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if dev.type == "cuda" else None,
                gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")


def finish(rd, a, model_name, model, sel_state, info, gtest_dl, parts, dev, extra):
    model.load_state_dict(sel_state)
    yg, pg = predict(model, gtest_dl, dev, a.amp)
    np.savez_compressed(rd / "global_test_preds.npz", y=yg, p=pg)
    info["global_test_selected"] = metrics(yg, pg)
    info.update(extra)
    if a.save_model: torch.save(sel_state, rd / "model_selected.pt")
    jdump(info, rd / "result.json")
    print(f"[done] {rd}  bal_acc={info['global_test_selected']['bal_acc']:.4f}  "
          f"t={info['wall_s'] / 60:.1f} min")


def stage_fed(a, dev):
    _, parts, gtest = load_manifest(a); K = len(parts)
    gdl = loader(gtest, a, False)
    vdl = [loader(p["val"], a, False) for p in parts]; tdl = [loader(p["test"], a, False) for p in parts]
    n_unique = np.array([len(p["train"]) for p in parts], float)
    w = n_unique / n_unique.sum()  # aggregation weights = UNIQUE training images per client
    for mname in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            rd = run_dir(a, "fed", mname, seed)
            if (rd / "result.json").exists(): print("[skip]", rd); continue
            rd.mkdir(parents=True, exist_ok=True); log = rd / "rounds.jsonl"
            if log.exists(): log.rename(rd / f"rounds_incomplete_{STAMP}.jsonl")
            set_seed(seed); t0 = time.time()
            model = build(mname, a.pretrained and mname != "secnn").to(dev)
            trl = [loader(p["train"], a, True, seed * 100 + k) for k, p in enumerate(parts)]
            gsd = cpu_sd(model)
            info = dict(kind="fed", algo="FedAvg (params + BN buffers averaged)", model=mname, seed=seed,
                        pretrained=bool(a.pretrained and mname != "secnn"), frozen_layers="none",
                        cam_layer=CAM_LAYER_NAMES[mname], rounds=a.rounds, local_epochs=a.local_epochs,
                        lr=a.lr, optimizer="Adam (reset each round)", loss="focal (gamma=2)", bs=a.bs,
                        img=a.img, vflip=a.vflip, aggregation_weights=w.tolist(),
                        n_train_unique=n_unique.astype(int).tolist(),
                        n_train_after_oversample=[len(d.sampler) for d in trl],
                        n_params=n_params(model), state_MB=state_bytes(model) / 2**20,
                        selection="round with max mean client-VALIDATION balanced accuracy",
                        env=env_info(dev))
            best, sel_state, sel_round = -1, None, 0
            for r in range(1, a.rounds + 1):
                sds = []
                for k in range(K):
                    model.load_state_dict(gsd)
                    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
                    train_epochs(model, trl[k], a, dev, a.local_epochs, opt); sds.append(cpu_sd(model))
                gsd = fedavg(sds, w); model.load_state_dict(gsd)
                vm = [metrics(*predict(model, d, dev, a.amp)) for d in vdl]
                cm = [metrics(*predict(model, d, dev, a.amp)) for d in tdl]
                gm = metrics(*predict(model, gdl, dev, a.amp))
                vscore = float(np.mean([m["bal_acc"] for m in vm]))
                jl(log, dict(round=r, val_mean_bal_acc=vscore, global_test=gm,
                             client_test=cm, client_val=[m["bal_acc"] for m in vm],
                             elapsed_s=time.time() - t0))
                if vscore > best: best, sel_state, sel_round = vscore, copy.deepcopy(gsd), r
                print(f"  [fed {mname} s{seed}] r{r}/{a.rounds} val={vscore:.4f} "
                      f"test_bal={gm['bal_acc']:.4f} worst_client={min(m['bal_acc'] for m in cm):.4f}")
            model.load_state_dict(sel_state)
            cm_sel = [metrics(*predict(model, d, dev, a.amp)) for d in tdl]
            finish(rd, a, mname, model, sel_state, info, gdl, parts, dev,
                   dict(selected_round=sel_round, best_val=best, client_test_selected=cm_sel,
                        wall_s=time.time() - t0))


def stage_central(a, dev):
    _, parts, gtest = load_manifest(a)
    gdl = loader(gtest, a, False)
    tr = pd.concat([p["train"] for p in parts]); va = pd.concat([p["val"] for p in parts])
    vdl = loader(va, a, False); tdl = [loader(p["test"], a, False) for p in parts]
    cvdl = [loader(p["val"], a, False) for p in parts]     # v6.2: harmonized selection
    E = a.rounds * a.local_epochs
    for mname in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            rd = run_dir(a, "central", mname, seed)
            if (rd / "result.json").exists(): print("[skip]", rd); continue
            rd.mkdir(parents=True, exist_ok=True); log = rd / "rounds.jsonl"
            if log.exists(): log.rename(rd / f"rounds_incomplete_{STAMP}.jsonl")
            set_seed(seed); t0 = time.time()
            model = build(mname, a.pretrained and mname != "secnn").to(dev)
            dl = loader(tr, a, True, seed); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            best, sel = -1, None
            for e in range(1, E + 1):
                train_epochs(model, dl, a, dev, 1, opt)
                v_pooled = metrics(*predict(model, vdl, dev, a.amp))["bal_acc"]
                v = float(np.mean([metrics(*predict(model, d, dev, a.amp))["bal_acc"] for d in cvdl]))
                gm = metrics(*predict(model, gdl, dev, a.amp))
                jl(log, dict(epoch=e, val_mean_client_bal_acc=v, val_pooled_bal_acc=v_pooled,
                             global_test=gm, elapsed_s=time.time() - t0))
                if v > best: best, sel, se = v, cpu_sd(model), e
                print(f"  [central {mname} s{seed}] e{e}/{E} val={v:.4f} test_bal={gm['bal_acc']:.4f}")
            model.load_state_dict(sel)
            cm = [metrics(*predict(model, d, dev, a.amp)) for d in tdl]
            info = dict(kind="central", model=mname, seed=seed, epochs=E, lr=a.lr, vflip=a.vflip,
                        selection="epoch with max mean client-VALIDATION balanced accuracy (as FedAvg)",
                        n_train_unique=len(tr), n_train_after_oversample=len(dl.sampler),
                        n_params=n_params(model), pretrained=bool(a.pretrained and mname != "secnn"),
                        env=env_info(dev))
            finish(rd, a, mname, model, sel, info, gdl, parts, dev,
                   dict(selected_epoch=se, best_val=best, client_test_selected=cm, wall_s=time.time() - t0))


def stage_local(a, dev):
    _, parts, gtest = load_manifest(a); gdl = loader(gtest, a, False)
    E = a.rounds * a.local_epochs
    for mname in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            rd = run_dir(a, "local", mname, seed)
            if (rd / "result.json").exists(): print("[skip]", rd); continue
            rd.mkdir(parents=True, exist_ok=True); t0 = time.time()
            if (rd / "global_test_preds.npz").exists():
                (rd / "global_test_preds.npz").rename(rd / f"global_test_preds_incomplete_{STAMP}.npz")
            own, glob, yg_all, pg_all = [], [], None, []
            for k, p in enumerate(parts):
                set_seed(seed * 100 + k)
                model = build(mname, a.pretrained and mname != "secnn").to(dev)
                dl = loader(p["train"], a, True, seed * 100 + k); vdl = loader(p["val"], a, False)
                opt = torch.optim.Adam(model.parameters(), lr=a.lr); best, sel = -1, None
                for e in range(1, E + 1):
                    train_epochs(model, dl, a, dev, 1, opt)
                    v = metrics(*predict(model, vdl, dev, a.amp))["bal_acc"]
                    if v > best: best, sel = v, cpu_sd(model)
                    print(f"  [local {mname} s{seed} c{k}] e{e}/{E} val={v:.4f}")
                model.load_state_dict(sel)
                own.append(metrics(*predict(model, loader(p["test"], a, False), dev, a.amp)))
                yg, pg = predict(model, gdl, dev, a.amp); glob.append(metrics(yg, pg)); pg_all.append(pg)
            np.savez_compressed(rd / "global_test_preds.npz", y=yg, p=np.mean(pg_all, 0))
            info = dict(kind="local", model=mname, seed=seed, epochs=E, vflip=a.vflip,
                        client_test_own_model=own, global_test_per_client_model=glob,
                        global_test_selected=dict(
                            bal_acc=float(np.mean([g["bal_acc"] for g in glob])),
                            acc=float(np.mean([g["acc"] for g in glob])),
                            sens=float(np.mean([g["sens"] for g in glob])),
                            spec=float(np.mean([g["spec"] for g in glob])),
                            macro_f1=float(np.mean([g["macro_f1"] for g in glob])),
                            auc=float(np.nanmean([g["auc"] for g in glob]))),
                        client_test_selected=own, wall_s=time.time() - t0, env=env_info(dev))
            jdump(info, rd / "result.json"); print("[done]", rd)


# --------------------------------------------------------------------------- analysis
def frs(bal, worst, sigma, w=(1 / 3, 1 / 3, 1 / 3), tau=0.10):
    """Exploratory Federated Reliability Score = wA*A + wW*W + wS*S,
    A = global-test balanced accuracy, W = worst-client test balanced accuracy,
    S = max(0, 1 - sigma/tau), sigma = SD of mean client-VALIDATION balanced accuracy over the last 5 rounds."""
    S = max(0.0, 1 - sigma / tau)
    return w[0] * bal + w[1] * worst + w[2] * S, S


def load_all(a):
    rows = []
    for f in sorted((Path(a.out) / "results").glob("*/*/seed*/result.json")):
        r = json.load(open(f)); exp = f.parts[-4]
        g = r["global_test_selected"]
        cl = r.get("client_test_selected") or []
        worst = min(c["bal_acc"] for c in cl) if cl else np.nan
        best_c = max(c["bal_acc"] for c in cl) if cl else np.nan
        sigma = np.nan; last_bal = np.nan; sigma_sel = np.nan
        log = f.parent / "rounds.jsonl"
        if log.exists() and exp.startswith("fed"):
            L = [json.loads(l) for l in open(log)]
            tail = [x["val_mean_bal_acc"] for x in L[-5:]]           # v6: stability on client VALIDATION
            sigma = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
            sr = r.get("selected_round") or len(L)                    # v6.2: window ending at selected round
            win = [x["val_mean_bal_acc"] for x in L[max(0, sr - 5):sr]]
            sigma_sel = float(np.std(win, ddof=1)) if len(win) > 1 else 0.0
            last_bal = L[-1]["global_test"]["bal_acc"]
        row = dict(exp=exp, model=r["model"], seed=r["seed"], **{f"g_{k}": g.get(k) for k in
                   ("acc", "bal_acc", "sens", "spec", "prec", "f1", "macro_f1", "auc", "tn", "fp", "fn", "tp")},
                   worst_client_bal=worst, gap_client_bal=best_c - worst, sigma_last5=sigma,
                   last_round_bal=last_bal, selected=r.get("selected_round", r.get("selected_epoch")),
                   n_params=r.get("n_params"), state_MB=r.get("state_MB"), wall_min=r["wall_s"] / 60,
                   gpu=(r.get("env") or {}).get("gpu"), path=str(f.parent), best_val=r.get("best_val"),
                   sigma_sel_window=sigma_sel if exp.startswith("fed") and log.exists() else np.nan)
        if exp.startswith("fed") and not np.isnan(sigma):
            row["FRS"], row["S"] = frs(row["g_bal_acc"], worst, sigma)
            row["FRS_selwin"], row["S_selwin"] = frs(row["g_bal_acc"], worst, sigma_sel)
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_exact(y, a_, b_):
    ca, cb = (a_ == y), (b_ == y); b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj


def bal_acc_vec(y, yh):
    s = ((yh == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    t = ((yh == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
    return (s + t) / 2


def stage_analyze(a):
    out = Path(a.out); an = out / "analysis_v1" / time.strftime("%Y%m%d_%H%M%S"); an.mkdir(parents=True, exist_ok=True)
    df = load_all(a)
    if df.empty: print("no results yet"); return
    df.to_csv(an / "per_seed_results_v1.csv", index=False)
    num = [c for c in df.columns if c.startswith("g_") or c in
           ("worst_client_bal", "gap_client_bal", "sigma_last5", "FRS", "FRS_selwin", "wall_min")]
    summ = df.groupby(["exp", "model"])[num].agg(["mean", "std"])
    summ.columns = [f"{a_}_{b}" for a_, b in summ.columns]
    summ = summ.reset_index().sort_values(["exp", "g_bal_acc_mean"], ascending=[True, False])
    summ.to_csv(an / "summary_mean_sd_v1.csv", index=False)

    # majority-class baseline on the global test set
    m = pd.read_csv(out / "split_manifest_v1.csv"); gt = m[m.partition == "global_test"].reset_index(drop=True)
    prev = gt.label.mean(); maj = int(prev >= 0.5)
    base = metrics(gt.label.values, np.full(len(gt), float(maj)))
    base.update(prevalence_pcos=prev, predicted_class=maj); jdump(base, an / "majority_baseline_v1.json")

    # paired statistics on the FED experiment (v6): reference chosen on VALIDATION, GROUP bootstrap
    fed = df[df.exp == "fed"]
    cis = []
    if len(fed):
        ref = fed.groupby("model").best_val.mean().idxmax()
        others = [x for x in fed.groupby("model").g_bal_acc.mean().sort_values(ascending=False).index if x != ref]
        grp = gt.group_id.values; ug = np.unique(grp); gidx = {g_: np.where(grp == g_)[0] for g_ in ug}
        rng = np.random.default_rng(a.split_seed)
        boots = [np.concatenate([gidx[g_] for g_ in rng.choice(ug, len(ug), replace=True)]) for _ in range(a.n_boot)]
        P = {}
        for _, r in fed.iterrows():
            z = np.load(Path(r.path) / "global_test_preds.npz"); P[(r.model, r.seed)] = (z["y"], z["p"])
        seeds = sorted(fed.seed.unique()); mcomp = len(others); alpha_b = 0.05 / max(1, mcomp)
        for other in others:
            common = [s_ for s_ in seeds if (ref, s_) in P and (other, s_) in P]
            if not common: continue
            diffs = np.zeros(a.n_boot); point = []
            for s_ in common:
                ya, pa = P[(ref, s_)]; yb, pb = P[(other, s_)]
                assert (ya == gt.label.values).all() and (yb == ya).all(), "prediction order mismatch"
                ha, hb = (pa >= .5).astype(int), (pb >= .5).astype(int)
                point.append(bal_acc_vec(ya, ha) - bal_acc_vec(yb, hb))
                diffs += np.array([bal_acc_vec(ya[i], ha[i]) - bal_acc_vec(ya[i], hb[i]) for i in boots])
            diffs /= len(common)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            blo, bhi = np.percentile(diffs, [100 * alpha_b / 2, 100 * (1 - alpha_b / 2)])
            cis.append(dict(ref_selected_on_validation=ref, vs=other, n_seeds=len(common),
                            mean_diff_bal_acc=float(np.mean(point)), ci95_lo=float(lo), ci95_hi=float(hi),
                            ci95_excludes_0=bool(lo > 0 or hi < 0),
                            bonferroni_level=1 - alpha_b, bonf_lo=float(blo), bonf_hi=float(bhi),
                            bonf_excludes_0=bool(blo > 0 or bhi < 0), n_groups=len(ug)))
        pd.DataFrame(cis).to_csv(an / "paired_group_bootstrap_ci_v2.csv", index=False)

        # paired per-client comparisons (same client-specific test sets, same seeds)
        pc = []
        for f in sorted((Path(a.out) / "results").glob("*/*/seed*/result.json")):
            r = json.load(open(f)); exp = f.parts[-4]
            if exp not in ("fed", "local", "central"): continue
            for k, c in enumerate(r.get("client_test_selected") or []):
                pc.append(dict(exp=exp, model=r["model"], seed=r["seed"], client=k, bal_acc=c["bal_acc"], n=c["n"]))
        pc = pd.DataFrame(pc)
        if len(pc):
            w_ = pc.pivot_table(index=["model", "seed", "client"], columns="exp", values="bal_acc").reset_index()
            w_.to_csv(an / "per_client_paired_v2.csv", index=False)
            if {"fed", "local"} <= set(w_.columns):
                s_ = w_.dropna(subset=["fed", "local"]).assign(diff=lambda x: x.fed - x.local)
                s_.groupby(["model", "client"])["diff"].agg(["mean", "min", "max"]).reset_index() \
                    .to_csv(an / "per_client_fed_minus_local_v2.csv", index=False)
                print(s_.groupby(["model", "client"])["diff"].agg(["mean", "min", "max"]).to_string())

        # FRS weight / threshold sensitivity (exploratory)
        sens = []
        grid_w = [(1/3, 1/3, 1/3), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5), (.6, .4, 0)]
        for w in grid_w:
            for tau in (0.05, 0.10, 0.20):
                s = fed.apply(lambda r: frs(r.g_bal_acc, r.worst_client_bal, r.sigma_last5, w, tau)[0], axis=1)
                tab = fed.assign(F=s).groupby("model").F.mean().sort_values(ascending=False)
                sens.append(dict(weights=str(tuple(round(x, 2) for x in w)), tau=tau,
                                 ranking=" > ".join(tab.index), top=tab.index[0]))
        pd.DataFrame(sens).to_csv(an / "frs_sensitivity_v1.csv", index=False)

    # compute cost
    cost = df.groupby(["exp", "model"]).agg(runs=("seed", "count"), wall_min_mean=("wall_min", "mean"),
                                            wall_min_total=("wall_min", "sum"), n_params=("n_params", "first"),
                                            state_MB=("state_MB", "first"), gpu=("gpu", "first")).reset_index()
    if "state_MB" in cost:
        cost["upload_MB_per_round_all_clients"] = cost.state_MB * 5
    cost.to_csv(an / "compute_cost_v1.csv", index=False)

    # markdown report
    def md(d):
        d = d.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + \
            "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)
    keep = ["exp", "model", "g_bal_acc_mean", "g_bal_acc_std", "g_sens_mean", "g_spec_mean",
            "g_macro_f1_mean", "g_auc_mean", "worst_client_bal_mean", "FRS_mean", "FRS_std"]
    keep = [k for k in keep if k in summ.columns]
    rep = ["# FedPCOS-XAI revision results (auto-generated)", "",
           f"Majority-class baseline on global test: prevalence(PCOS)={prev:.4f}, "
           f"bal_acc={base['bal_acc']:.4f}, macro_F1={base['macro_f1']:.4f}", "",
           "## Mean +- SD over seeds", "", md(summ[keep]), "",
           "## Per-seed results", "", md(df[["exp", "model", "seed", "g_bal_acc", "g_sens", "g_spec", "g_auc",
                                             "worst_client_bal", "selected", "g_tn", "g_fp", "g_fn", "g_tp"]]), ""]
    if cis: rep += ["## Paired GROUP bootstrap CI of balanced-accuracy difference (reference chosen on validation)", "",
                    md(pd.DataFrame(cis)), ""]
    rep += ["## Compute cost", "", md(cost), ""]
    (an / "REPORT_v1.md").write_text("\n".join(rep))
    print("\n".join(rep[:12])); print(f"\nWritten to {an}")


# --------------------------------------------------------------------------- XAI
class GradCAM:
    def __init__(s, model, layer):
        s.m = model; s.act = None; s.grad = None
        layer.register_forward_hook(s._fh)
    def _fh(s, mod, i, o):
        s.act = o
        if o.requires_grad: o.register_hook(lambda g: setattr(s, "grad", g))
    def __call__(s, x, cls=1):
        s.m.zero_grad(set_to_none=True); x = x.clone().requires_grad_(True)
        out = s.m(x); out[:, cls].sum().backward()
        w = s.grad.mean((2, 3), keepdim=True); cam = F.relu((w * s.act).sum(1, keepdim=True))
        s.grid = tuple(cam.shape[-2:])
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True); cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach(), F.softmax(out, 1).detach()


def no_inplace(m):
    for mod in m.modules():
        if hasattr(mod, "inplace"): mod.inplace = False
    return m


@torch.no_grad()
def deletion_auc(model, x, sal, cls, steps=10):
    order = torch.argsort(sal.flatten(), descending=True); n = order.numel(); sc = []
    for i in range(steps + 1):
        xm = x.clone(); k = int(n * i / steps)
        if k:
            mask = torch.ones(n, device=x.device); mask[order[:k]] = 0
            xm = xm * mask.view(1, 1, *sal.shape)
        sc.append(F.softmax(model(xm), 1)[0, cls].item())
    sc = np.asarray(sc); return float(((sc[1:] + sc[:-1]) / 2).sum() / steps)


def spearman(a_, b_):
    r = stats.spearmanr(a_.flatten().cpu().numpy(), b_.flatten().cpu().numpy()).correlation
    return float(r) if r == r else float("nan")


def stage_xai(a, dev):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(a.out); xd = out / "xai_v1" / time.strftime("%Y%m%d_%H%M%S"); xd.mkdir(parents=True, exist_ok=True)
    _, _, gt = load_manifest(a)
    seeds = [int(s) for s in a.seeds.split(",")]
    models = [mm for mm in a.models.split(",") if (run_dir(a, "fed", mm, seeds[0]) / "model_selected.pt").exists()]
    if not models: print("[xai] no saved fed models"); return
    # prespecified sample: fixed random per class + errors of the best model (seed 0)
    df = load_all(a); fed = df[df.exp == "fed"]
    best = fed.groupby("model").best_val.mean().idxmax()          # v6: reference chosen on validation
    z = np.load(run_dir(a, "fed", best, seeds[0]) / "global_test_preds.npz")
    gt = gt.reset_index(drop=True); gt["pred_best"] = (z["p"] >= .5).astype(int)
    rnd = pd.concat([gt[gt.label == c].sample(min(int((gt.label == c).sum()), a.xai_per_class),
                                              random_state=a.split_seed) for c in (0, 1)])
    err = gt[gt.pred_best != gt.label]
    err = err.sample(min(len(err), a.xai_max_errors), random_state=a.split_seed)
    samp = pd.concat([rnd, err]).drop_duplicates("path")
    samp["subset"] = np.where(samp.index.isin(err.index), "error_of_best", "random")
    samp.to_csv(xd / "xai_sample_manifest_v1.csv", index=False)
    tf = tfm(a, False); rows = []
    X = [tf(Image.open(img_path(a.data_root, p)).convert("RGB")) for p in samp.path]
    panel = {}
    for mname in models:
        def load(seed):
            m = no_inplace(build(mname, False)).to(dev)
            m.load_state_dict(torch.load(run_dir(a, "fed", mname, seed) / "model_selected.pt", map_location=dev))
            return m.eval()
        m0 = load(seeds[0]); cam0 = GradCAM(m0, cam_layer(m0, mname))
        m1 = load(seeds[1]) if len(seeds) > 1 and (run_dir(a, "fed", mname, seeds[1]) / "model_selected.pt").exists() else None
        cam1 = GradCAM(m1, cam_layer(m1, mname)) if m1 else None
        set_seed(12345); mr = no_inplace(build(mname, False)).to(dev).eval(); camr = GradCAM(mr, cam_layer(mr, mname))
        for i, (x, (_, r)) in enumerate(zip(X, samp.iterrows())):
            x = x[None].to(dev)
            c_p, pr = cam0(x, 1)                         # PCOS-class map
            pred = int(pr[0, 1] >= .5)
            c_pred, _ = cam0(x, pred)                    # predicted-class map
            row = dict(model=mname, path=r.path, label=int(r.label), pred=pred, subset=r.subset,
                       p_pcos=float(pr[0, 1]))
            g = torch.Generator(device="cpu").manual_seed(1000 + i)
            rnd_sal = torch.rand(c_p[0].shape, generator=g).to(dev)                  # pixel-level random order
            gh, gw = cam0.grid                                                         # CAM native grid
            low = torch.rand((1, 1, gh, gw), generator=g).to(dev)                      # matched: same grid, same upsampling
            blob_sal = F.interpolate(low, c_p[0].shape, mode="bilinear", align_corners=False)[0, 0]
            for tag, sal, cls in (("pcos", c_p[0], 1), ("pred", c_pred[0], pred)):
                const = bool(float(sal.max() - sal.min()) < 1e-6)
                row[f"{tag}_map_constant"] = const
                # deletion monitors the SAME fixed class the map explains; masked pixels are set to 0
                # AFTER ImageNet normalization (i.e. replaced by the ImageNet mean colour)
                row[f"{tag}_del_auc_cam"] = np.nan if const else deletion_auc(m0, x, sal, cls)
                row[f"{tag}_del_auc_random"] = deletion_auc(m0, x, rnd_sal, cls)
                row[f"{tag}_del_auc_matched"] = deletion_auc(m0, x, blob_sal, cls)
                row[f"{tag}_diff_vs_matched"] = row[f"{tag}_del_auc_cam"] - row[f"{tag}_del_auc_matched"]
            row["pcos_cam_beats_random"] = (row["pcos_del_auc_cam"] < row["pcos_del_auc_random"]) \
                if not row["pcos_map_constant"] else np.nan
            row["pred_cam_beats_random"] = (row["pred_del_auc_cam"] < row["pred_del_auc_random"]) \
                if not row["pred_map_constant"] else np.nan
            cr, _ = camr(x, 1)
            rconst = bool(float(cr[0].max() - cr[0].min()) < 1e-6)
            row["randomized_map_constant"] = rconst
            row["randomization_spearman"] = np.nan if (rconst or row["pcos_map_constant"]) else spearman(c_p[0], cr[0])
            if cam1:
                c1, _ = cam1(x, 1); row["seed_stability_spearman"] = spearman(c_p[0], c1[0])
            rows.append(row)
            if i < 6: panel.setdefault(i, {})[mname] = c_p[0].cpu().numpy()
        print(f"[xai] {mname} done")
    res = pd.DataFrame(rows); res.to_csv(xd / "xai_per_image_v1.csv", index=False)
    def bci(v):
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        if not len(v): return (np.nan, np.nan, np.nan, 0)
        b = np.random.default_rng(0).choice(v, (2000, len(v))).mean(1)
        return (v.mean(), *np.percentile(b, [2.5, 97.5]), len(v))
    ag = []
    for (mname, lab), g in [((mm, "all"), res[res.model == mm]) for mm in res.model.unique()] + \
                           [((mm, int(l)), res[(res.model == mm) & (res.label == l)]) for mm in res.model.unique() for l in (0, 1)]:
        row = dict(model=mname, label=lab, n=len(g))
        for col in ("pcos_del_auc_cam", "pcos_del_auc_random", "pcos_del_auc_matched", "pcos_diff_vs_matched",
                    "pcos_cam_beats_random",
                    "pred_del_auc_cam", "pred_del_auc_random", "pred_del_auc_matched", "pred_diff_vs_matched",
                    "pred_cam_beats_random",
                    "randomization_spearman", "seed_stability_spearman"):
            if col in g:
                mu, lo, hi, n_ = bci(g[col].astype(float))
                row.update({col: mu, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_n": n_})
        row["n_randomization_nonevaluable"] = int(g.randomization_spearman.isna().sum())
        ag.append(row)
    agg = pd.DataFrame(ag)
    agg.to_csv(xd / "xai_summary_v1.csv", index=False); print(agg.to_string(index=False))
    # overlay figure: 6 images x models
    inv = lambda t: np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
    fig, ax = plt.subplots(len(panel), len(models) + 1, figsize=(2.2 * (len(models) + 1), 2.2 * len(panel)))
    ax = np.atleast_2d(ax)
    for i in panel:
        ax[i, 0].imshow(inv(X[i])); ax[i, 0].set_title(f"label={int(samp.iloc[i].label)}", fontsize=8)
        for j, mname in enumerate(models):
            ax[i, j + 1].imshow(inv(X[i])); ax[i, j + 1].imshow(panel[i][mname], cmap="jet", alpha=.4)
            if i == 0: ax[i, j + 1].set_title(mname, fontsize=8)
    for x_ in ax.flat: x_.axis("off")
    plt.tight_layout(); plt.savefig(xd / "gradcam_panel_v1.png", dpi=150); plt.close()


# --------------------------------------------------------------------------- v6.2: CNN shortcut reliance
def _manip(x, kind, raw_wh=None, frame=0.12, centre=0.5):
    """x: normalized 1x3xHxW tensor. Masked pixels are set to 0 (ImageNet mean colour)."""
    H, W = x.shape[-2:]; y = x.clone()
    if kind == "original": return y
    fh, fw = int(H * frame), int(W * frame)
    if kind == "border_removed":                          # remove outer frame (text, scale bars, overlays, padding)
        m = torch.zeros_like(y); m[..., fh:H - fh, fw:W - fw] = 1; return y * m
    ch, cw = int(H * (1 - centre) / 2), int(W * (1 - centre) / 2)
    if kind == "centre_only":                             # keep only the central region
        m = torch.zeros_like(y); m[..., ch:H - ch, cw:W - cw] = 1; return y * m
    if kind == "periphery_only":                          # remove the central region, keep the periphery
        m = torch.ones_like(y); m[..., ch:H - ch, cw:W - cw] = 0; return y * m
    raise ValueError(kind)


def stage_shortcutcnn(a, dev):
    """Scores the trained FEDERATED models (all models, all seeds, 224 px) on manipulated global-test inputs."""
    out = Path(a.out); sd = out / "shortcutcnn_v2" / time.strftime("%Y%m%d_%H%M%S"); sd.mkdir(parents=True, exist_ok=True)
    _, _, gt = load_manifest(a); gt = gt.reset_index(drop=True)
    tf_norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    def load_img(p, pad):
        im = Image.open(img_path(a.data_root, p)).convert("RGB")
        if pad:                                           # aspect-preserving: pad to square with black, then resize
            w, h = im.size; s_ = max(w, h); c = Image.new("RGB", (s_, s_)); c.paste(im, ((s_ - w) // 2, (s_ - h) // 2)); im = c
        return tf_norm(im.resize((a.img, a.img)))
    X = torch.stack([load_img(p, False) for p in gt.path]); XP = torch.stack([load_img(p, True) for p in gt.path])
    y = gt.label.values.astype(int); rows = []
    kinds = ["original", "border_removed", "centre_only", "periphery_only", "pad_square"]
    for mname in a.models.split(","):
        for seed in [int(s_) for s_ in a.seeds.split(",")]:
            f = run_dir(a, "fed", mname, seed) / "model_selected.pt"
            if not f.exists(): continue
            m = build(mname, False).to(dev); m.load_state_dict(torch.load(f, map_location=dev)); m.eval()
            for kind in kinds:
                src = XP if kind == "pad_square" else X; ps = []
                with torch.no_grad():
                    for i in range(0, len(src), 64):
                        xb = src[i:i + 64].to(dev)
                        if kind not in ("original", "pad_square"): xb = torch.cat([_manip(xb[j:j + 1], kind) for j in range(len(xb))])
                        ps.append(F.softmax(m(xb).float(), 1)[:, 1].cpu().numpy())
                p = np.concatenate(ps); mt = metrics(y, p)
                rows.append(dict(model=mname, seed=seed, input=kind, bal_acc=mt["bal_acc"], sens=mt["sens"],
                                 spec=mt["spec"], auc=mt["auc"]))
            print(f"[shortcutcnn] {mname} s{seed} done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(sd / "shortcut_cnn_per_seed_v2.csv", index=False)
    s_ = r.groupby(["model", "input"]).bal_acc.agg(["mean", "std"]).unstack("input")
    s_.to_csv(sd / "shortcut_cnn_summary_v2.csv"); print(s_.round(4).to_string())


# --------------------------------------------------------------------------- main
def run_all(a, dev):
    """Full one-day queue in priority order (resumable, nothing overwritten)."""
    seeds = a.seeds
    def step(title, **kw):
        b = copy.copy(a); b.tag = ""; b.vflip = 0; b.seeds = seeds
        for k, v in kw.items(): setattr(b, k, v)
        print(f"\n[{time.strftime('%H:%M')}] ===== {title} =====", flush=True)
        {"regroup": stage_regroup, "split": stage_split, "leakaudit": stage_leakaudit,
         "shortcut": stage_shortcut, "analyze": stage_analyze}[b.stage](b) \
            if b.stage in ("regroup", "split", "leakaudit", "shortcut", "analyze") else \
            {"fed": stage_fed, "central": stage_central, "local": stage_local, "xai": stage_xai,
             "shortcutcnn": stage_shortcutcnn}[b.stage](b, dev)
    step("0a) regroup (transformation-aware groups, ~5 min CPU)", stage="regroup")
    step("0b) split", stage="split")
    step("0c) leakage audit by partition", stage="leakaudit")
    step("0d) metadata shortcut probes", stage="shortcut")
    step("1) MAIN: 8 models x 3 seeds, FedAvg", stage="fed",
         models="secnn,densenet121,resnet50,googlenet", amp=1)
    step("1b) MAIN: MobileNetV2/ShuffleNetV2 in FP32 (FP16 corrupts BN buffers)", stage="fed",
         models="mobilenetv2,shufflenetv2", amp=0)
    step("1c) MAIN: AlexNet/VGG-16 at lr 1e-4", stage="fed", models="alexnet,vgg16", lr=1e-4)
    step("2) CENTRALIZED baseline", stage="central", models="secnn,densenet121")
    step("2b) CENTRALIZED VGG-16 at lr 1e-4", stage="central", models="vgg16", lr=1e-4)
    step("3) LOCAL-ONLY baseline", stage="local", models="secnn,densenet121")
    step("4) VERTICAL-FLIP ablation", stage="fed", models="secnn,densenet121", vflip=1, tag="_vflip")
    step("4b) FAILED-CONFIG demo: MobileNetV2/ShuffleNetV2 in FP16", stage="fed",
         models="mobilenetv2,shufflenetv2", amp=1, tag="_fp16")
    step("4c) FAILED-CONFIG demo: AlexNet/VGG-16 at lr 1e-3", stage="fed",
         models="alexnet,vgg16", lr=1e-3, tag="_lr1e3")
    step("interim analysis", stage="analyze")
    step("5) XAI audit", stage="xai", models=",".join(ALL_MODELS))
    step("6) CNN shortcut-reliance evaluation", stage="shortcutcnn", models=",".join(ALL_MODELS))
    step("final analysis", stage="analyze")
    print(f"\nALL DONE -> newest folders in {a.out}/analysis_v1/ and {a.out}/xai_v1/")


def main():
    a = args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.stage not in ("regroup", "split", "leakaudit", "shortcut", "analyze") and dev.type != "cuda":
        print("WARNING: CUDA not available - running on CPU (very slow).")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jdump(vars(a), Path(a.out) / "args" / f"{a.stage}{a.tag}_{STAMP}.json")
    if a.stage in ("all", "regroup", "fed", "central", "local", "xai", "shortcutcnn"): preflight(a)
    if a.stage == "all": return run_all(a, dev)
    {"regroup": lambda: stage_regroup(a), "leakaudit": lambda: stage_leakaudit(a),
     "shortcut": lambda: stage_shortcut(a),
     "split": lambda: stage_split(a), "fed": lambda: stage_fed(a, dev),
     "central": lambda: stage_central(a, dev), "local": lambda: stage_local(a, dev),
     "analyze": lambda: stage_analyze(a), "xai": lambda: stage_xai(a, dev),
     "shortcutcnn": lambda: stage_shortcutcnn(a, dev)}[a.stage]()


if __name__ == "__main__":
    main()
