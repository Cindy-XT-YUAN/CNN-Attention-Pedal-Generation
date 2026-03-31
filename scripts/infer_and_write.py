
# 推理 + 回写踏板CC64，生成新的 MIDI
# 依赖：pip install pretty_midi mido numpy torch
"""
CUDA_VISIBLE_DEVICES=0 python ./scripts/infer_and_write.py \
  --input ./*.mid \
  --ckpt ./runs/exp_cnn_attn_ddp/pedal_best.pth \
  --out ./*.mid \
  --fs 20 --on 0.6 --off 0.4 \
  --min_down_ms 150 --min_up_ms 80 --repedal_ms 30 \
  --hidden 256 --heads 8 --device cuda
"""
import os
import argparse
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pretty_midi as pm
import torch.nn.functional as F

# 模型,与训练时一致
class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, p=None, dropout=0.0):
        super().__init__()
        if p is None:
            p = d  
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, dilation=d, padding=p, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PedalModel4CNNAttn(nn.Module):
    def __init__(self, hidden=256, heads=8):
        super().__init__()
        # 4 层卷积，每层 256 通道，kernel=3, padding=1
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

        self.proj = nn.Conv1d(256, hidden, kernel_size=1)

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
        if x.dim() != 3:
            raise RuntimeError(f"expect [B,T,128] or [B,128,T], got {x.shape}")
        B, A, C = x.shape
        if C == 128:
            x = x.permute(0, 2, 1)   
            B, C, T_orig = x.shape
        elif A == 128:
            B, C, T_orig = x.shape
        else:
            raise RuntimeError(f"last dim must be 128, got {x.shape}")

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.proj(x)             
        x = x.permute(0, 2, 1)       
        x = self.norm1(x)
        x_t = x.transpose(0, 1)      

        attn_out, _ = self.attn(x_t, x_t, x_t, need_weights=False)
        x = x + attn_out.transpose(0, 1) 

        x = self.norm2(x)
        x = x + self.ffn(x)                
        y_small = torch.sigmoid(self.out(x).squeeze(-1)) 

        # 线性插值回原始 T
        y = F.interpolate(y_small.unsqueeze(1), size=T_orig, mode="linear", align_corners=True).squeeze(1)
        return y




# 音频/MIDI 处理工具
def get_pianoroll(midi: pm.PrettyMIDI, fs: int = 20) -> np.ndarray:
    #返回 [128, T] 的钢琴卷帘（0..1
    pr = midi.get_piano_roll(fs=fs)  
    pr = np.clip(pr, 0, 127) / 127.0
    return pr.astype(np.float32)

def triangular_window(L: int) -> np.ndarray:
    #用于重叠平均的三角窗，减小拼接边界的接缝
    if L <= 1:
        return np.ones((L,), dtype=np.float32)
    # 0 ... 1 ... 0
    w = 1.0 - np.abs((np.arange(L) - (L - 1) / 2) / ((L + 1) / 2))
    return w.astype(np.float32)

def overlap_add_avg(chunks: List[np.ndarray], starts: List[int], total_len: int, win: np.ndarray) -> np.ndarray:
    #将一堆长度相同的预测块（[Tseg]）按起点拼回全长 [T]，
    #使用加权平均（三角窗）平滑重叠区域。
    
    out = np.zeros((total_len,), dtype=np.float32)
    wsum = np.zeros((total_len,), dtype=np.float32)
    L = len(win)
    for pred, st in zip(chunks, starts):
        ed = st + L
        p = pred[:L]
        if ed > total_len:
            # 超尾部时截断窗口以对齐
            valid = total_len - st
            if valid <= 0:
                continue
            out[st:total_len] += p[:valid] * win[:valid]
            wsum[st:total_len] += win[:valid]
        else:
            out[st:ed] += p * win
            wsum[st:ed] += win
    # 防止除零
    mask = wsum > 0
    out[mask] /= wsum[mask]
    return out

# 后处理：滞回 + 最短段 + re-pedal
def hysteresis(x: np.ndarray, on_th: float = 0.55, off_th: float = 0.45) -> np.ndarray:
    #把 0..1 的概率序列 -> 二值开关序列（0/1），带滞回
    y = np.zeros_like(x, dtype=np.int32)
    state = 0
    for i, v in enumerate(x):
        if state == 0 and v >= on_th:
            state = 1
        elif state == 1 and v <= off_th:
            state = 0
        y[i] = state
    return y

def enforce_min_segments(b: np.ndarray, min_down: int, min_up: int) -> np.ndarray:
    #基于运行长度编码，消除过短的踩/抬段
    if b.size == 0:
        return b
    out = b.copy()
    i = 0
    n = len(out)
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        seg_len = j - i
        val = out[i]
        if val == 1 and seg_len < min_down:
            out[i:j] = 0
        elif val == 0 and seg_len < min_up:
            out[i:j] = 1
        i = j
    return out

def enforce_repedal_gap(b: np.ndarray, gap: int) -> np.ndarray:
    #确保抬起后至少 gap 帧才允许再次踩下
    if gap <= 0 or len(b) == 0:
        return b
    out = b.copy()
    last_up_idx = None
    for i in range(1, len(out)):
        if out[i-1] == 1 and out[i] == 0:
            last_up_idx = i
        if last_up_idx is not None and out[i] == 1:
            # 若距离上次抬起不足 gap，则强制保持为 0
            if i - last_up_idx < gap:
                out[i] = 0
    return out

def binary_to_cc_events(b: np.ndarray, fs: int) -> List[Tuple[float, int]]:
    #二值序列转 CC64 事件 (time_sec, value 0/127)
    events: List[Tuple[float, int]] = []
    if len(b) == 0:
        return events
    cur = b[0]
    if cur == 1:
        events.append((0.0, 127))
    for t in range(1, len(b)):
        if b[t] != cur:
            cur = b[t]
            events.append((t / fs, 127 if cur == 1 else 0))
    return events

# 主逻辑：推理 & 回写
def infer_pedal_for_midi(midi_in: str,
                         ckpt: str,
                         midi_out: str = None,
                         fs: int = 20,
                         seg_len_s: float = 4.0,
                         hop_s: float = 2.0,
                         on_th: float = 0.55,
                         off_th: float = 0.45,
                         min_down_ms: int = 180,
                         min_up_ms: int = 100,
                         repedal_ms: int = 40,
                         hidden: int = 96,
                         heads: int = 4,
                         device: str = "auto",
                         batch_size: int = 64,
                         remove_existing: bool = True,
                         write_all_tracks: bool = True) -> str:
    
    #读取无踏板 MIDI -> 模型预测 -> 后处理 -> 写回 CC64 -> 保存新 MIDI
    if midi_out is None:
        root, ext = os.path.splitext(midi_in)
        midi_out = root + "_with_pedal.mid"

    # 设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 载入 MIDI & 生成钢琴卷帘
    midi = pm.PrettyMIDI(midi_in)
    pr = get_pianoroll(midi, fs=fs)  # [128,T]
    T = pr.shape[1]
    if T == 0:
        raise RuntimeError("MIDI 时长为 0，无法推理。")

    # 切片
    Tseg = int(round(seg_len_s * fs))
    Thop = int(round(hop_s * fs))
    starts = list(range(0, T, Thop))
    # 三角窗
    win = triangular_window(Tseg)

    # 组 batch 推理
    model = PedalModel4CNNAttn(hidden=hidden, heads=heads).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    preds_chunks: List[np.ndarray] = []
    with torch.no_grad():
        batch = []
        batch_idx = []
        for st in starts:
            ed = st + Tseg
            if ed <= T:
                x = pr[:, st:ed]
            else:
                # 末尾不足一段，用 0 补齐
                pad = ed - T
                x = np.pad(pr[:, st:T], ((0, 0), (0, pad)), mode="constant")
            batch.append(x)            
            batch_idx.append(st)
            if len(batch) == batch_size:
                X = np.stack(batch, axis=0)                
                X = torch.tensor(X, dtype=torch.float32, device=device)
                pred = model(X).cpu().numpy()               
                preds_chunks.extend(list(pred))
                batch.clear(); batch_idx.clear()
        if batch:
            X = np.stack(batch, axis=0)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            pred = model(X).cpu().numpy()
            preds_chunks.extend(list(pred))

    # 重叠-平均 拼回全长
    pred_full = overlap_add_avg(preds_chunks, starts, total_len=T, win=win)  

    # 后处理
    b = hysteresis(pred_full, on_th=on_th, off_th=off_th) 
    min_down = max(1, int(round(min_down_ms / 1000.0 * fs)))
    min_up   = max(1, int(round(min_up_ms   / 1000.0 * fs)))
    gap      = max(0, int(round(repedal_ms  / 1000.0 * fs)))

    b = enforce_min_segments(b, min_down=min_down, min_up=min_up)
    b = enforce_repedal_gap(b, gap=gap)

    # 转换为 CC64 事件 (0/127)
    events = binary_to_cc_events(b, fs=fs)  

    # 写回
    target_instruments = [inst for inst in midi.instruments if not inst.is_drum]
    if not target_instruments:
        # 若没有旋律乐器，创建一个钢琴轨
        inst = pm.Instrument(program=0, name="piano")
        midi.instruments.append(inst)
        target_instruments = [inst]

    for inst in target_instruments if write_all_tracks else [target_instruments[0]]:
        if remove_existing:
            inst.control_changes = [cc for cc in inst.control_changes if cc.number != 64]
        for t, val in events:
            inst.control_changes.append(pm.ControlChange(number=64, value=int(val), time=float(t)))
        # 排序
        inst.control_changes.sort(key=lambda c: c.time)

    midi.write(midi_out)
    return midi_out


def parse_args():
    ap = argparse.ArgumentParser(description="Infer sustain pedal (CC64) and write back to MIDI.")
    ap.add_argument("--input", required=True, help="输入 MIDI 文件（建议无踏板或忽略原踏板）")
    ap.add_argument("--ckpt",  required=True, help="模型权重路径，例如 runs/pedal_best.pth")
    ap.add_argument("--out",   default=None, help="输出 MIDI 文件，默认 <input>_with_pedal.mid")

    # 与训练一致的参数
    ap.add_argument("--fs", type=int, default=20, help="帧率Hz，训练/推理需一致（默认20）")
    ap.add_argument("--seg_len", type=float, default=4.0, help="窗口秒数（默认4.0）")
    ap.add_argument("--hop",     type=float, default=2.0, help="步长秒数（默认2.0）")

    # 后处理参数
    ap.add_argument("--on",  type=float, default=0.55, help="滞回上阈值（默认0.55）")
    ap.add_argument("--off", type=float, default=0.45, help="滞回下阈值（默认0.45）")
    ap.add_argument("--min_down_ms", type=int, default=180, help="最短踩下时长ms（默认180）")
    ap.add_argument("--min_up_ms",   type=int, default=100, help="最短抬起时长ms（默认100）")
    ap.add_argument("--repedal_ms",  type=int, default=40,  help="re-pedal 间隙ms（默认40）")

    # 模型与运行
    ap.add_argument("--hidden", type=int, default=96, help="隐藏通道数，应与训练一致")
    ap.add_argument("--heads",  type=int, default=4,  help="注意力头数，应与训练一致")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    ap.add_argument("--batch_size", type=int, default=64, help="推理批大小")

    # 写回策略
    ap.add_argument("--keep_old", action="store_true", help="保留原有 CC64（默认删除）")
    ap.add_argument("--first_track_only", action="store_true", help="仅写回到第一条非鼓轨（默认写回所有非鼓轨）")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = infer_pedal_for_midi(
        midi_in=args.input,
        ckpt=args.ckpt,
        midi_out=args.out,
        fs=args.fs,
        seg_len_s=args.seg_len,
        hop_s=args.hop,
        on_th=args.on,
        off_th=args.off,
        min_down_ms=args.min_down_ms,
        min_up_ms=args.min_up_ms,
        repedal_ms=args.repedal_ms,
        hidden=args.hidden,
        heads=args.heads,
        device=args.device,
        batch_size=args.batch_size,
        remove_existing=(not args.keep_old),
        write_all_tracks=(not args.first_track_only),
    )
    print("✅ Saved:", out)
