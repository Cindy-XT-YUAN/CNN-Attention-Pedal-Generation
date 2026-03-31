"""
Usage:
  python plot_eval_core.py \
    --csv eval_out_full/eval_per_clip.csv \
    --out_dir figs --prefix full_ --dpi 300
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to eval_per_clip.csv")
    ap.add_argument("--out_dir", default="figs", help="Directory to save figures")
    ap.add_argument("--prefix", default="", help="Filename prefix for outputs")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    ap.add_argument("--figw", type=float, default=7.0, help="Figure width (inches)")
    ap.add_argument("--figh", type=float, default=5.0, help="Figure height (inches)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.csv)

    required_cols = {
        "true_time_ratio", "pred_time_ratio",
        "true_dur_ms_median", "pred_dur_ms_median"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少列: {missing}. "
                         f"请确认这是 eval_pedal_dist.py 生成的 eval_per_clip.csv。")

    # 图1：踏板时间占比散点图 
    plt.figure(figsize=(args.figw, args.figh))
    plt.scatter(df["true_time_ratio"], df["pred_time_ratio"], s=10, alpha=0.5)
    # 参考对角线
    lo = min(df["true_time_ratio"].min(), df["pred_time_ratio"].min(), 0.0)
    hi = max(df["true_time_ratio"].max(), df["pred_time_ratio"].max(), 1.0)
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True pedal time ratio")
    plt.ylabel("Predicted pedal time ratio")
    plt.title("Pedal Time Ratio per Clip (Pred vs. True)")
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(args.out_dir, f"{args.prefix}ratio_scatter.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=args.dpi)
    print(f"Saved: {out1}")

    # 图2：单次踏板中位时长直方图（预测 vs 真实
    plt.figure(figsize=(args.figw, args.figh))
    plt.hist(df["pred_dur_ms_median"], bins=60, alpha=0.6, label="Pred", density=False)
    plt.hist(df["true_dur_ms_median"], bins=60, alpha=0.6, label="True", density=False)
    plt.xlabel("Median pedal duration per clip (ms)")
    plt.ylabel("Count")
    plt.title("Median Pedal Duration Distribution (Pred vs. True)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(args.out_dir, f"{args.prefix}duration_hist.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=args.dpi)
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()
