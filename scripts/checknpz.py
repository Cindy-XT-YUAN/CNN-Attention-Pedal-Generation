import numpy as np, json, sys, os

npz_path = sys.argv[1] if len(sys.argv) > 1 else "data/maestro_processed.npz"
meta_path = os.path.splitext(npz_path)[0] + "_meta.json"

print("=> Loading:", npz_path)
d = np.load(npz_path)
X, Y = d["features"], d["labels"]
print("features:", X.shape, "labels:", Y.shape)  # 期望 [N,T,128], [N,T]

# 基本一致性
assert X.ndim == 3 and Y.ndim == 2, "维度应为 X:[N,T,128], Y:[N,T]"
N, T, C = X.shape
assert C == 128, f"最后一维应为128音高，got {C}"
assert Y.shape[0] == N and Y.shape[1] == T, f"T不一致: X.T={T}, Y.T={Y.shape[1]}"

print("=> OK: shapes match. N =", N, "T =", T)

# 数值范围
print("X range:", float(X.min()), "→", float(X.max()), "(should be within [0,1])")
print("Y range:", float(Y.min()), "→", float(Y.max()), "(should be within [0,1])")

# 统计踏板占比与事件粗略数量（基于阈值0.5）
on = (Y >= 0.5).astype(np.float32)
ratio = on.mean()
events = 0
for i in range(N):
    s = on[i]
    # 统计 0->1 上升沿个数
    events += int(((s[1:] > s[:-1]) & (s[1:] == 1)).sum())
print(f"Pedal on-ratio (>=0.5): {ratio:.3f}, approx events per clip: {events/N:.1f}")
if os.path.exists(meta_path):
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    print("meta:", json.dumps(meta, ensure_ascii=False)[:300], "...")
else:
    print("(meta json not found:", meta_path, ")")
