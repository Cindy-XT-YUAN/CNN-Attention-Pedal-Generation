import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_log", type=str, required=True, help="Path to training csv log file")
    parser.add_argument("--out", type=str, default="loss_curve.png", help="Output image file")
    args = parser.parse_args()

    # 读取日志
    df = pd.read_csv(args.csv_log)

    # 检查列名
    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        raise ValueError("CSV 文件必须包含 'epoch', 'train_loss', 'val_loss' 三列")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"曲线已保存到 {args.out}")

if __name__ == "__main__":
    main()
