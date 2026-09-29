"""Figure 6 (v18): PCOS-class Grad-CAM of the eight federated models (seed 42) on
3 PCOS + 2 non-PCOS images (random audit sample) + 1 image misclassified by the reference model.
Run inside the container, in /workspace (same folder as fedpcos_pipeline.py and runs_v2):
    python figure6_mixed_v18.py
Output: runs_v2/figure6_v18/Figure6_gradcam_mixed_v18.png / .tiff and figure6_images_v18.csv
"""
import sys, glob, types, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("pipe", "fedpcos_pipeline.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

OUT = Path("runs_v2"); DATA = "/workspace/data/PCOS_figshare_ds"
FD = OUT / "figure6_v18"; FD.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = ["secnn", "alexnet", "vgg16", "resnet50", "densenet121", "googlenet", "mobilenetv2", "shufflenetv2"]
NAME = dict(secnn="SE-CNN", alexnet="AlexNet", vgg16="VGG-16", resnet50="ResNet-50", densenet121="DenseNet-121",
            googlenet="GoogLeNet", mobilenetv2="MobileNetV2", shufflenetv2="ShuffleNetV2")

sm = pd.read_csv(sorted(glob.glob(str(OUT / "xai_v1" / "*" / "xai_sample_manifest_v1.csv")))[-1])
rnd = sm[sm.subset == "random"]
pick = pd.concat([rnd[rnd.label == 1].head(3), rnd[rnd.label == 0].head(2), sm[sm.subset != "random"].head(1)]).reset_index(drop=True)
pick.to_csv(FD / "figure6_images_v18.csv", index=False)

a = types.SimpleNamespace(img=224, out=str(OUT), tag="", vflip=0)
tf = P.tfm(a, False)
X = [tf(Image.open(P.img_path(DATA, p)).convert("RGB")) for p in pick.path]
inv = lambda t: np.clip(t.permute(1, 2, 0).numpy() * np.array(P.STD) + np.array(P.MEAN), 0, 1)

fig, ax = plt.subplots(len(pick), len(M) + 1, figsize=(2.1 * (len(M) + 1), 2.25 * len(pick)))
for j, m in enumerate(M):
    net = P.no_inplace(P.build(m, False)).to(dev)
    net.load_state_dict(torch.load(P.run_dir(a, "fed", m, 42) / "model_selected.pt", map_location=dev)); net.eval()
    cam = P.GradCAM(net, P.cam_layer(net, m))
    for i, x in enumerate(X):
        c, pr = cam(x[None].to(dev), 1)
        a_ = ax[i, j + 1]; a_.imshow(inv(x)); a_.imshow(c[0].cpu().numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        a_.set_xlabel(f"P(PCOS) = {pr[0, 1].item():.2f}", fontsize=8); a_.set_xticks([]); a_.set_yticks([])
    print(m, "done", flush=True)
for i, r in pick.iterrows():
    a_ = ax[i, 0]; a_.imshow(inv(X[i])); a_.set_xticks([]); a_.set_yticks([])
    lab = "PCOS" if r.label == 1 else "non-PCOS"
    a_.set_ylabel(lab + ("\n(misclassified by\nreference model)" if r.subset != "random" else ""), fontsize=9)
for j, t in enumerate(["Input"] + [NAME[m] for m in M]): ax[0, j].set_title(t, fontsize=10, fontweight="bold")
fig.tight_layout()
for ext in ("png", "tiff"):
    fig.savefig(FD / f"Figure6_gradcam_mixed_v18.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print("Saved to", FD)
