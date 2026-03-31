"""
# 1) 评估 best 模型（GPU）
python eval_pedal_distribution.py \
  --npz data/maestro_processed.npz \
  --ckpt runs/exp_cnn_attn_ddp/pedal_best.pth \
  --out_dir eval_out_best \
  --device cuda \
  --batch_size 256 \
  --hidden 256 --heads 8 \
  --fs 20 --on 0.60 --off 0.40 \
  --min_down_ms 150 --min_up_ms 80 --repedal_ms 200

# 2) 只抽样前 5000 段快速评估
python eval_pedal_distribution.py \
  --npz data/maestro_processed.npz \
  --ckpt runs/exp_cnn_attn_ddp/pedal_best.pth \
  --out_dir eval_out_sample \
  --device cuda \
  --limit 5000



"""

import os
import json
import math
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_device(x, device):
    return x.to(device, non_blocking=True)

def upsample_linear(x: torch.Tensor, out_T: int) -> torch.Tensor:
    B, Tin = x.shape
    if Tin == out_T:
        return x
    x = x.unsqueeze(1) 
    x = F.interpolate(x, size=out_T, mode="linear", align_corners=True)
    return x.squeeze(1) 

def hysteresis_binarize(prob: np.ndarray, on=0.6, off=0.4):
    """Hysteresis to reduce flicker: on when >=on, off when <=off."""
    out = np.zeros_like(prob, dtype=np.uint8)
    down = False
    for i, v in enumerate(prob):
        if not down and v >= on:
            down = True
        elif down and v <= off:
            down = False
        out[i] = 1 if down else 0
    return out

def events_from_binary(b: np.ndarray, fs=20, min_down_ms=150, min_up_ms=80):
    """Return list of (start_idx, end_idx) after debouncing."""
    ev = []
    i, n = 0, len(b)
    while i < n:
        if b[i] == 1:
            j = i + 1
            while j < n and b[j] == 1:
                j += 1
            dur_ms = (j - i) * 1000.0 / fs
            if dur_ms >= min_down_ms:
                ev.append((i, j))
            i = j
        else:
            i += 1

    if not ev:
        return ev
    merged = [ev[0]]
    for s, e in ev[1:]:
        prev_s, prev_e = merged[-1]
        gap_ms = (s - prev_e) * 1000.0 / fs
        if gap_ms < min_up_ms:
            # merge
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged

def summarize_events(ev, fs=20, total_T=None, repedal_ms=200):
    durs_ms = [(e1 - s1) * 1000.0 / fs for s1, e1 in ev]
    gaps_ms = []
    for k in range(1, len(ev)):
        gap = (ev[k][0] - ev[k-1][1]) * 1000.0 / fs
        gaps_ms.append(gap)
    repedal_rate = float(np.mean([g < repedal_ms for g in gaps_ms])) if gaps_ms else 0.0
    total_down = float(sum(durs_ms))
    ratio = None
    if total_T is not None:
        total_ms = total_T * 1000.0 / fs
        ratio = 0.0 if total_ms <= 0 else total_down / total_ms
    def pctl(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0
    return dict(
        count=int(len(ev)),
        dur_ms_median=pctl(durs_ms, 50),
        dur_ms_p10=pctl(durs_ms, 10),
        dur_ms_p90=pctl(durs_ms, 90),
        gap_ms_median=pctl(gaps_ms, 50),
        repedal_rate=repedal_rate,
        time_ratio=0.0 if ratio is None else float(ratio)
    )

# Metrics
def frame_metrics(y_true, y_prob):
    """
    y_true, y_prob: 1D numpy arrays of same length, y_true in [0,1], y_prob in [0,1]
    Returns MSE, Brier, AUROC, AUPRC (AU* may fail if labels single-class).
    """
    from math import isfinite
    mse = float(np.mean((y_prob - y_true) ** 2))
    # Brier 对二值标签定义；这里将 true>=0.5 视为正类
    y_bin = (y_true >= 0.5).astype(np.uint8)
    brier = float(np.mean((y_prob - y_bin) ** 2))

    auroc = None
    auprc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # 需要同时有正负样本
        if y_bin.max() != y_bin.min():
            auroc = float(roc_auc_score(y_bin, y_prob))
            auprc = float(average_precision_score(y_bin, y_prob))
    except Exception:
        auroc = None
        auprc = None

    return dict(MSE=mse, Brier=brier, AUROC=auroc, AUPRC=auprc)

def calibration_bins(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    y_bin = (y_true >= 0.5).astype(np.uint8)
    out = []
    for b in range(n_bins):
        mask = (idx == b)
        cnt = int(mask.sum())
        if cnt == 0:
            out.append(dict(bin=f"[{bins[b]:.2f},{bins[b+1]:.2f})", count=0, pred_mean=None, true_rate=None))
            continue
        pred_mean = float(y_prob[mask].mean())
        true_rate = float(y_bin[mask].mean())
        out.append(dict(bin=f"[{bins[b]:.2f},{bins[b+1]:.2f})", count=cnt, pred_mean=pred_mean, true_rate=true_rate))
    return out

# Model (should match your training architecture)
class PedalModel4CNNAttn(nn.Module):
    def __init__(self, hidden=256, heads=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.proj  = nn.Conv1d(256, hidden, kernel_size=1)

        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden, hidden*4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden*4, hidden),
        )

        self.out   = nn.Linear(hidden, 1)

    def forward(self, x): 
        if x.size(-1) == 128:  
            x = x.permute(0, 2, 1)
        B, C, T_orig = x.shape

        x = self.conv1(x); x = self.pool1(x)     
        x = self.conv2(x); x = self.pool2(x)   
        x = self.conv3(x); x = self.conv4(x)     

        x = self.proj(x)                            
        x = x.permute(0, 2, 1)                     
        x = self.norm1(x)
        x_t = x.transpose(0, 1)                  

        attn_out, _ = self.attn(x_t, x_t, x_t, need_weights=False)
        x = x + attn_out.transpose(0, 1)           
        x = self.norm2(x)
        x = x + self.ffn(x)                       

        y_small = torch.sigmoid(self.out(x).squeeze(-1))  
        y = F.interpolate(y_small.unsqueeze(1), size=T_orig, mode="linear", align_corners=True).squeeze(1)
        return y



# Main evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",  required=True, help="path to maestro_processed.npz")
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint (pedal_best.pth)")
    ap.add_argument("--out_dir", default="eval_out", help="output directory")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads",  type=int, default=8)

    ap.add_argument("--fs", type=int, default=20)
    ap.add_argument("--on",  type=float, default=0.60)
    ap.add_argument("--off", type=float, default=0.40)
    ap.add_argument("--min_down_ms", type=int, default=150)
    ap.add_argument("--min_up_ms",   type=int, default=80)
    ap.add_argument("--repedal_ms",  type=int, default=200)
    ap.add_argument("--calib_bins",  type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="limit number of clips to evaluate (0 = all)")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    device = torch.device("cuda" if (args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available())) else "cpu")

    print(f"=> Loading dataset: {args.npz}")
    data = np.load(args.npz)
    X = data["features"]  
    Y = data["labels"]    
    N, T, D = X.shape
    assert D == 128, f"features last dim must be 128, got {D}"
    print(f"features: {X.shape} labels: {Y.shape}")

    if args.limit and args.limit > 0:
        N_lim = min(N, args.limit)
        X = X[:N_lim]
        Y = Y[:N_lim]
        N = N_lim
        print(f"=> Limited to first {N} clips")

    # Build model
    model = PedalModel4CNNAttn(hidden=args.hidden, heads=args.heads).to(device)
    ckpt_path = Path(args.ckpt)
    print(f"=> Loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e}\n       retry with strict=False")
        model.load_state_dict(state, strict=False)
    model.eval()

    # Inference loop
    bs = args.batch_size
    y_pred_all = np.zeros_like(Y, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.from_numpy(X[i:i+bs]).float()
            xb = to_device(xb, device)
            yb = model(xb)       
            y_pred_all[i:i+bs] = yb.detach().cpu().numpy()

    y_true_flat = Y.reshape(-1).astype(np.float32)
    y_prob_flat = y_pred_all.reshape(-1).astype(np.float32)
    frame_stat  = frame_metrics(y_true_flat, y_prob_flat)
    frame_calib = calibration_bins(y_true_flat, y_prob_flat, n_bins=args.calib_bins)
    with open(out_dir / "eval_frame_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": frame_stat, "calibration": frame_calib}, f, ensure_ascii=False, indent=2)
    print(f"=> Saved frame metrics to {out_dir/'eval_frame_metrics.json'}")

    fs = args.fs
    per_clip_rows = []
    pred_time_ratios, true_time_ratios = [], []
    pred_event_counts, true_event_counts = [], []
    pred_dur_medians, true_dur_medians = [], []
    pred_gap_medians, true_gap_medians = [], []

    for idx in range(N):
        prob = y_pred_all[idx]
        true = Y[idx]

        pred_bin  = hysteresis_binarize(prob, on=args.on, off=args.off)
        true_bin  = (true >= 0.5).astype(np.uint8)  

        pred_ev = events_from_binary(pred_bin, fs=fs, min_down_ms=args.min_down_ms, min_up_ms=args.min_up_ms)
        true_ev = events_from_binary(true_bin, fs=fs, min_down_ms=args.min_down_ms, min_up_ms=args.min_up_ms)

        pred_sum = summarize_events(pred_ev, fs=fs, total_T=len(prob), repedal_ms=args.repedal_ms)
        true_sum = summarize_events(true_ev, fs=fs, total_T=len(true), repedal_ms=args.repedal_ms)

        per_clip_rows.append([
            idx,
            pred_sum["time_ratio"], true_sum["time_ratio"],
            pred_sum["count"],      true_sum["count"],
            pred_sum["dur_ms_median"], true_sum["dur_ms_median"],
            pred_sum["gap_ms_median"], true_sum["gap_ms_median"],
            pred_sum["repedal_rate"],  true_sum["repedal_rate"]
        ])

        pred_time_ratios.append(pred_sum["time_ratio"])
        true_time_ratios.append(true_sum["time_ratio"])
        pred_event_counts.append(pred_sum["count"])
        true_event_counts.append(true_sum["count"])
        pred_dur_medians.append(pred_sum["dur_ms_median"])
        true_dur_medians.append(true_sum["dur_ms_median"])
        pred_gap_medians.append(pred_sum["gap_ms_median"])
        true_gap_medians.append(true_sum["gap_ms_median"])

    csv_path = out_dir / "eval_per_clip.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "clip_id",
            "pred_time_ratio", "true_time_ratio",
            "pred_event_count","true_event_count",
            "pred_dur_ms_median","true_dur_ms_median",
            "pred_gap_ms_median","true_gap_ms_median",
            "pred_repedal_rate","true_repedal_rate"
        ])
        wr.writerows(per_clip_rows)
    print(f"=> Saved per-clip stats to {csv_path}")

    def agg(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return dict(mean=float(np.mean(arr)),
                    median=float(np.median(arr)),
                    p10=float(np.percentile(arr,10)),
                    p90=float(np.percentile(arr,90)))
    event_summary = dict(
        settings=dict(fs=fs, on=args.on, off=args.off,
                      min_down_ms=args.min_down_ms, min_up_ms=args.min_up_ms,
                      repedal_ms=args.repedal_ms),
        time_ratio=dict(pred=agg(pred_time_ratios), true=agg(true_time_ratios)),
        event_count=dict(pred=agg(pred_event_counts), true=agg(true_event_counts)),
        dur_ms_median=dict(pred=agg(pred_dur_medians), true=agg(true_dur_medians)),
        gap_ms_median=dict(pred=agg(pred_gap_medians), true=agg(true_gap_medians)),
    )
    with open(out_dir / "eval_event_summary.json", "w", encoding="utf-8") as f:
        json.dump(event_summary, f, ensure_ascii=False, indent=2)
    print(f"=> Saved event summary to {out_dir/'eval_event_summary.json'}")

    print("Done.")

if __name__ == "__main__":
    main()
