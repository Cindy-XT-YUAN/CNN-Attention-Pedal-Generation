import os, json, math, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#  Utilities
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_npz(npz_path, limit=None):
    data = np.load(npz_path, allow_pickle=False)
    X = data["features"].astype(np.float32)  
    Y = data["labels"].astype(np.float32)    
    if limit is not None:
        X = X[:limit]
        Y = Y[:limit]
    return X, Y


class PedalModel4CNNAttn(nn.Module):
    def __init__(self, c1=256, c2=256, c3=256, c4=256,
                 hidden=256, heads=8, dropout=0.0, ffn_hidden=None):
        super().__init__()
        C_in = 128  

        self.conv1 = nn.Sequential(
            nn.Conv1d(C_in, c1, 3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(c2, c3, 3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(c3, c4, 3, padding=1),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
        )

        self.proj  = nn.Conv1d(c4, hidden, 1)

        self.attn  = nn.MultiheadAttention(hidden, num_heads=heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(hidden)

        if ffn_hidden is None:
            ffn_hidden = hidden * 4  
        self.ffn   = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

        self.out   = nn.Linear(hidden, 1)

    def forward(self, x):  
        x = x.permute(0,2,1)         
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.proj(x)             
        x = x.permute(2,0,1)          

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        y = self.out(x).squeeze(-1)  
        y = y.permute(1,0)            
        return torch.sigmoid(y)

def infer_model_shapes_from_state(state):
    def out_ch(sd, k):
        return sd[k].shape[0] if k in sd else None

    c1 = out_ch(state, "conv1.0.weight")
    c2 = out_ch(state, "conv2.0.weight")
    c3 = out_ch(state, "conv3.0.weight")
    c4 = out_ch(state, "conv4.0.weight")
    if c1 is None: c1 = 256
    if c2 is None: c2 = c1
    if c3 is None: c3 = c2
    if c4 is None: c4 = c3

    ffn_hidden = state["ffn.0.weight"].shape[0] if "ffn.0.weight" in state else None
    return c1, c2, c3, c4, ffn_hidden

def hysteresis_debounce(prob, on=0.6, off=0.4, fs=20, min_down_ms=150, min_up_ms=80, repedal_ms=30):
    """
    prob: (T,) in [0,1]
    returns binarized array in {0,1} with min durations & optional repedal fuse.
    """
    T = len(prob)
    s = 0
    binv = np.zeros(T, dtype=np.uint8)
    for i, p in enumerate(prob):
        if s == 0 and p >= on: s = 1
        elif s == 1 and p <= off: s = 0
        binv[i] = s

    ms_per_frame = 1000.0 / fs
    min_down = max(1, int(round(min_down_ms / ms_per_frame)))
    min_up   = max(1, int(round(min_up_ms   / ms_per_frame)))
    repedal  = max(0, int(round(repedal_ms  / ms_per_frame)))

    def clamp_segments(arr, val, min_len):
        i = 0
        Tn = len(arr)
        while i < Tn:
            if arr[i] == val:
                j = i
                while j < Tn and arr[j] == val: j += 1
                if (j - i) < min_len:
                    arr[i:j] = 1 - val
                i = j
            else:
                i += 1
        return arr

    binv = clamp_segments(binv, 1, min_down)
    binv = clamp_segments(binv, 0, min_up)

    if repedal > 0:
        i = 0
        while i < T:
            if binv[i] == 0:
                j = i
                while j < T and binv[j] == 0: j += 1
                if (j - i) <= repedal:
                    binv[i:j] = 1
                i = j
            else:
                i += 1

    return binv

def pedal_stats_from_binary(binv, fs):
    """Return ratio (mean of 1s) and median on-duration (ms) for a clip."""
    ratio = binv.mean() if len(binv) > 0 else 0.0
    segs = []
    i = 0
    T = len(binv)
    while i < T:
        if binv[i] == 1:
            j = i
            while j < T and binv[j] == 1: j += 1
            segs.append(j - i)
            i = j
        else:
            i += 1
    if len(segs) == 0:
        med_ms = 0.0
    else:
        ms_per_frame = 1000.0 / fs
        med_ms = float(np.median(np.array(segs) * ms_per_frame))
    return ratio, med_ms

def kl_divergence(p_hist, q_hist, eps=1e-9):
    p = p_hist.astype(np.float64)
    q = q_hist.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))

def batched_predict(model, device, X, batch=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch), desc="Infer"):
            xb = torch.from_numpy(X[i:i+batch]).to(device)  
            pb = model(xb).cpu().numpy()                     
            preds.append(pb)
    return np.concatenate(preds, axis=0)

def score_config(pred_ratios, true_ratios, pred_durs, true_durs,
                 target_mean, target_std,
                 mean_lo, mean_hi, min_ratio_std,
                 w_ratio_kl=1.0, w_dur_kl=1.0, w_mean_mse=0.5, w_std_mse=0.5):
    pred_ratios = np.asarray(pred_ratios); true_ratios = np.asarray(true_ratios)
    pred_durs   = np.asarray(pred_durs);   true_durs   = np.asarray(true_durs)

    rhist_bins = np.linspace(0, 1, 41)
    dhist_bins = np.linspace(0, 2000, 41)
    pr_hist, _ = np.histogram(pred_ratios, bins=rhist_bins)
    tr_hist, _ = np.histogram(true_ratios, bins=rhist_bins)
    pd_hist, _ = np.histogram(pred_durs,   bins=dhist_bins)
    td_hist, _ = np.histogram(true_durs,   bins=dhist_bins)

    kl_ratio = kl_divergence(pr_hist, tr_hist)
    kl_dur   = kl_divergence(pd_hist, td_hist)

    mean_mse = (pred_ratios.mean() - target_mean) ** 2
    std_mse  = (pred_ratios.std()  - target_std ) ** 2

    score = (w_ratio_kl * kl_ratio +
             w_dur_kl   * kl_dur   +
             w_mean_mse * mean_mse +
             w_std_mse  * std_mse)

    collapse = False
    if pred_ratios.std() < min_ratio_std:
        collapse = True
    if not (mean_lo <= pred_ratios.mean() <= mean_hi):
        collapse = True
    if collapse:
        score += 100.0
    return float(score)

def evaluate_config(probs, labels, fs, on, off, min_down_ms, min_up_ms, repedal_ms):
    N, T = probs.shape
    true_ratios, pred_ratios = [], []
    true_durs,   pred_durs   = [], []
    for i in range(N):
        prob = probs[i]
        true = labels[i]
        bin_true = (true >= 0.5).astype(np.uint8)
        t_ratio, t_med = pedal_stats_from_binary(bin_true, fs)

        bin_pred = hysteresis_debounce(prob, on=on, off=off, fs=fs,
                                       min_down_ms=min_down_ms, min_up_ms=min_up_ms, repedal_ms=repedal_ms)
        p_ratio, p_med = pedal_stats_from_binary(bin_pred, fs)

        true_ratios.append(t_ratio); pred_ratios.append(p_ratio)
        true_durs.append(t_med);     pred_durs.append(p_med)
    return np.array(true_ratios), np.array(pred_ratios), np.array(true_durs), np.array(pred_durs)

def save_plots(true_ratios, pred_ratios, true_durs, pred_durs, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib not available, skip plots:", e)
        return
    out_dir = Path(out_dir)

    plt.figure(figsize=(8,6))
    plt.scatter(true_ratios, pred_ratios, s=6, alpha=0.35)
    lims=[0,1]; plt.plot(lims, lims, 'k--', lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True pedal time ratio")
    plt.ylabel("Predicted pedal time ratio")
    plt.title("Pedal Time Ratio per Clip (Pred vs. True)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_ratio.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8,6))
    bins = np.linspace(0, 2000, 50)
    plt.hist(pred_durs, bins=bins, alpha=0.7, label="Pred")
    plt.hist(true_durs, bins=bins, alpha=0.7, label="True")
    plt.xlabel("Median pedal duration per clip (ms)")
    plt.ylabel("Count")
    plt.title("Median Pedal Duration Distribution (Pred vs. True)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_duration.png", dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser("Auto-tune post-processing with collapse guards")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--max_trials", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--fs", type=int, default=20)

    ap.add_argument("--on_range",  nargs=2, type=float, default=[0.45, 0.70])
    ap.add_argument("--off_range", nargs=2, type=float, default=[0.30, 0.55])
    ap.add_argument("--min_down_ms_range", nargs=2, type=int, default=[80, 220])
    ap.add_argument("--min_up_ms_range",   nargs=2, type=int, default=[40, 150])
    ap.add_argument("--repedal_ms_range",  nargs=2, type=int, default=[10, 80])

    ap.add_argument("--target_mean", type=float, default=0.55)
    ap.add_argument("--target_std",  type=float, default=0.22)
    ap.add_argument("--min_ratio_std", type=float, default=0.05)
    ap.add_argument("--mean_lo",       type=float, default=0.15)
    ap.add_argument("--mean_hi",       type=float, default=0.85)

    args = ap.parse_args()
    set_seed(1337)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = load_npz(args.npz, limit=args.limit)
    N, T, F = X.shape
    print(f"=> Dataset: N={N}, T={T}, F={F}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device)
    c1, c2, c3, c4, ffn_hidden = infer_model_shapes_from_state(state)
    model = PedalModel4CNNAttn(c1=c1, c2=c2, c3=c3, c4=c4,
                               hidden=256, heads=8, dropout=0.0,
                               ffn_hidden=ffn_hidden).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print("[warn] strict load failed, but shapes inferred; retry strict=False:", e)
        model.load_state_dict(state, strict=False)
    model.eval()

    probs = batched_predict(model, device, X, batch=args.batch) 
    labels = Y  

    bin_true = (labels >= 0.5).astype(np.uint8)
    true_ratios = bin_true.mean(axis=1)
    true_durs = []
    for i in range(N):
        _, t_med = pedal_stats_from_binary(bin_true[i], fs=args.fs)
        true_durs.append(t_med)
    true_durs = np.array(true_durs)

    def U(a, b, integer=False):
        v = random.uniform(a, b)
        return int(round(v)) if integer else float(v)

    best = {"score": 1e9, "cfg": None, "pred_ratios": None, "pred_durs": None}
    print(f"=> Try {args.max_trials} configs")
    pbar = tqdm(total=args.max_trials, desc="Grid")
    for _ in range(args.max_trials):
        on  = U(*args.on_range)
        off = U(*args.off_range)
        if on <= off: 
            on = min(0.99, off + 0.05)
        cfg = dict(
            on=round(on,3),
            off=round(off,3),
            min_down_ms=U(*args.min_down_ms_range, integer=True),
            min_up_ms=U(*args.min_up_ms_range, integer=True),
            repedal_ms=U(*args.repedal_ms_range, integer=True)
        )

        tr, pr, td, pd = evaluate_config(
            probs, labels, fs=args.fs,
            on=cfg["on"], off=cfg["off"],
            min_down_ms=cfg["min_down_ms"], min_up_ms=cfg["min_up_ms"], repedal_ms=cfg["repedal_ms"]
        )

        score = score_config(
            pr, tr, pd, td,
            target_mean=args.target_mean, target_std=args.target_std,
            mean_lo=args.mean_lo, mean_hi=args.mean_hi, min_ratio_std=args.min_ratio_std,
            w_ratio_kl=1.0, w_dur_kl=1.0, w_mean_mse=0.5, w_std_mse=0.5
        )

        if score < best["score"]:
            best = {"score": score, "cfg": cfg, "pred_ratios": pr, "pred_durs": pd}

        pbar.set_postfix(best=f"{best['score']:.3f}", on=cfg["on"], off=cfg["off"])
        pbar.update(1)
    pbar.close()

    result = {
        "best_score": best["score"],
        "best_cfg": best["cfg"],
        "pred_mean_ratio": float(np.mean(best["pred_ratios"])),
        "pred_std_ratio":  float(np.std(best["pred_ratios"]))
    }
    with open(out_dir / "best_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("=> Best:", json.dumps(result, indent=2, ensure_ascii=False))

    save_plots(true_ratios, best["pred_ratios"], true_durs, best["pred_durs"], out_dir)

if __name__ == "__main__":
    main()
