import pretty_midi as pm
import argparse
import os
from glob import glob

def analyze_file(path):
    try:
        midi = pm.PrettyMIDI(path)
    except Exception as e:
        return {"path": path, "error": str(e)}

    length = midi.get_end_time() or 1e-9
    has_cc64 = False
    total_pedal_time = 0.0

    for inst in midi.instruments:
        # 过滤鼓
        if inst.is_drum:
            continue
        cc64 = [cc for cc in inst.control_changes if cc.number == 64]
        if not cc64:
            continue

        has_cc64 = True
        cc64.sort(key=lambda c: c.time)
        # 估算总"踩下"时长（阈值>=64 视为踩下）
        down = False
        last_t = 0.0
        ped_time = 0.0
        for cc in cc64:
            if not down and cc.value >= 64:
                down = True
                last_t = cc.time
            elif down and cc.value < 64:
                down = False
                ped_time += max(0.0, cc.time - last_t)

        # 结尾仍踩下则补到曲末
        if down:
            ped_time += max(0.0, length - last_t)

        total_pedal_time += ped_time

    return {
        "has_pedal": has_cc64,
        "length_s": length,
        "total_pedal_time_s": total_pedal_time
    }

def main():
    ap = argparse.ArgumentParser(description="Check if MIDI contains sustain pedal (CC64).")
    ap.add_argument("input", help="MIDI file path or a directory to scan")
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.input):
        for ext in ("*.mid", "*.midi"):
            paths += glob(os.path.join(args.input, "**", ext), recursive=True)
        if not paths:
            print("No MIDI files found in directory.")
            return
    else:
        paths = [args.input]

    total_files = len(paths)
    files_with_pedal = 0
    total_length = 0.0
    total_pedal_time = 0.0

    for p in paths:
        r = analyze_file(p)
        if "error" in r:
            print(f"[ERROR] {p}: {r['error']}")
            total_files -= 1
            continue

        if r["has_pedal"]:
            files_with_pedal += 1
            total_length += r["length_s"]
            total_pedal_time += r["total_pedal_time_s"]

    # 计算比例
    pedal_file_ratio = files_with_pedal / total_files if total_files > 0 else 0
    pedal_time_ratio = total_pedal_time / total_length if total_length > 0 else 0

    # 输出最终结果
    print(f"Total files: {total_files}")
    print(f"Files with CC64: {files_with_pedal} ({pedal_file_ratio:.2%})")
    print(f"Total length: {total_length:.2f} s")
    print(f"Total pedal time: {total_pedal_time:.2f} s ({pedal_time_ratio:.2%})")

if __name__ == "__main__":
    main()
