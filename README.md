
# Data
Figshare PCOS ultrasound archive: https://figshare.com/articles/dataset/PCOS_Dataset/27682557
Expected layout: `PCOS_figshare_ds/PCOS/infected` and `PCOS_figshare_ds/PCOS/noninfected`.
The file inventory (`file_inventory.csv`, SHA-256 exact-duplicate status) and the split manifests with checksums are supplied in the review package.

## Environment
Python 3.12.3, PyTorch 2.12.0, torchvision 0.27.0, CUDA 13.2, cuDNN 9.22 (NVIDIA PyTorch container 26.05), one NVIDIA RTX A4000 (16 GB).

```
pip install -r requirements.txt
python fedpcos_pipeline.py --stage all --inventory file_inventory.csv --data_root /path/to/PCOS_figshare_ds --out runs_v2
```
Seeds 42, 7 and 99; 5 clients; Dirichlet alpha = 2.0; 20 rounds x 2 local epochs; batch size 32; 224 px input.
Run `python fedpcos_pipeline.py --help` for all options.

## Figures
`figures/make_figures.py` (Figures 1-5, 7) and `figures/figure6_gradcam.py` (Figure 6) read the outputs under `runs_v2/`.
