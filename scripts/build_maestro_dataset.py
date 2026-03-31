#!/usr/bin/env python3
# pip install pretty_midi mido tqdm numpy

import os, sys, math, json, argparse
from glob import glob
import numpy as np
import pretty_midi as pm
from tqdm import tqdm

def get_pianoroll(midi: pm.PrettyMIDI, fs: int = 20) -> np.ndarray:
    pr = midi.get_piano_roll(fs=fs)  # [128, T], 0..127
    pr = np.clip(pr, 0, 127) / 127.0
    return pr

#def get_cc64_track(midi: pm.PrettyMIDI, fs: int = 20) -> np.ndarray:
#    T = max(1, int(math.ceil(midi.get_end_time() * fs)))
#    cc64 = np.zeros(T, dtype=np.float32)
#    for inst in midi.instruments:
#        if inst.is_drum:
#            continue
#        for cc in inst.control_changes:
#            if cc.number == 64:
#                t = min(T - 1, max(0, int(cc.time * fs)))
#                cc64[t:] = np.maximum(cc64[t:], cc.value / 127.0)
#    return cc64
def get_cc64_track(midi, fs=20):
    T = max(1, int(np.ceil(midi.get_end_time() * fs)))
    track = np.zeros(T, dtype=np.float32)
    # 收集所有非鼓轨的 CC64 事件
    evts = []
    for inst in midi.instruments:
        if inst.is_drum: continue
        for cc in inst.control_changes:
            if cc.number == 64:
                evts.append((cc.time, cc.value))
    if not evts:
        return track  # 全 0（无踏板）

    evts.sort(key=lambda x: x[0])
    last_val = 0.0
    last_i = 0
    for t, v in evts:
        i = min(T-1, max(0, int(t*fs)))
        # 从上一个事件到当前事件，保持 last_val
        track[last_i:i] = last_val
        last_val = v/127.0
        last_i = i
    track[last_i:] = last_val
    return track


def slice_segments(X_128xT: np.ndarray, y_T: np.ndarray, fs: int, seg_len_s: float, hop_s: float):
    T = X_128xT.shape[1]
    Tseg = int(round(seg_len_s * fs))
    Thop = int(round(hop_s * fs))
    xs, ys, starts = [], [], []
    for start in range(0, max(1, T - Tseg + 1), Thop):
        end = start + Tseg
        if end > T:
            pad = end - T
            x_seg = np.pad(X_128xT[:, start:T], ((0,0),(0,pad)))
            y_seg = np.pad(y_T[start:T], (0, pad))
        else:
            x_seg = X_128xT[:, start:end]
            y_seg = y_T[start:end]
        xs.append(x_seg.T.astype(np.float32))  # [Tseg,128]
        ys.append(y_seg.astype(np.float32))    # [Tseg]
        starts.append(start / fs)
    return xs, ys, starts

def collect_midi_files(root: str):
    files = []
    for ext in ("*.mid", "*.midi"):
        files += glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(description="Build maestro_processed.npz from MAESTRO MIDI")
    ap.add_argument("--midi_root", required=True, help="MAESTRO MIDI 根目录（包含10个子目录/1200+文件）")
    ap.add_argument("--out_npz", default="maestro_processed.npz")
    ap.add_argument("--out_meta", default="maestro_meta.json")
    ap.add_argument("--fs", type=int, default=20, help="帧率Hz（默认20=每帧50ms）")
    ap.add_argument("--seg_len", type=float, default=4.0, help="片段秒数（默认4s）")
    ap.add_argument("--hop", type=float, default=2.0, help="滑窗步长秒数（默认2s）")
    ap.add_argument("--min_note_activity", type=float, default=0.0,
                    help="过滤空白片段：帧均值阈值（0..1），默认不过滤")
    args = ap.parse_args()

    files = collect_midi_files(args.midi_root)
    if not files:
        print("未找到任何 MIDI 文件，请检查 --midi_root"); sys.exit(1)

    X_list, Y_list, meta = [], [], []
    for f in tqdm(files, desc="Processing MIDI"):
        try:
            midi = pm.PrettyMIDI(f)
            if midi.get_end_time() <= 0: continue
        except Exception as e:
            print(f"[WARN] 跳过: {f} ({e})"); continue

        pr = get_pianoroll(midi, fs=args.fs)     # [128,T]
        cc64 = get_cc64_track(midi, fs=args.fs)  # [T]

        xs, ys, starts = slice_segments(pr, cc64, args.fs, args.seg_len, args.hop)

        if args.min_note_activity > 0:
            kept = [(x,y,s) for (x,y,s) in zip(xs,ys,starts) if float(x.mean()) >= args.min_note_activity]
            if kept:
                xs, ys, starts = zip(*kept)
            else:
                xs, ys, starts = [], [], []

        for x,y,s in zip(xs,ys,starts):
            X_list.append(x); Y_list.append(y); meta.append({"file": f, "start_sec": round(s,3)})

    if not X_list:
        print("没有得到任何片段，请调小 min_note_activity 或检查参数"); sys.exit(2)

    X = np.stack(X_list, axis=0)  # [N,T,128], 0..1
    Y = np.stack(Y_list, axis=0)  # [N,T],     0..1

    np.savez_compressed(args.out_npz, features=X, labels=Y)
    with open(args.out_meta, "w", encoding="utf-8") as w:
        json.dump({"fs": args.fs, "seg_len": args.seg_len, "hop": args.hop,
                   "features_shape": list(X.shape), "labels_shape": list(Y.shape),
                   "segments": meta}, w, ensure_ascii=False, indent=2)

    print(f"✅ 保存完成：{args.out_npz} | features {X.shape} labels {Y.shape}")
    print(f"📝 元数据：{args.out_meta}")

if __name__ == "__main__":
    main()

