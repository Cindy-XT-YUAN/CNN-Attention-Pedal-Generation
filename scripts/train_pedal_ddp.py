"""
train_pedal_ddp.py
- 从 maestro_processed.npz 读取 [N,T,128] 特征 和 [N,T] 标签（0..1）
- 模型：4×1D CNN（dilation=1,1,2,4）+ MultiheadAttention（batch_first）
- DDP 训练：torchrun 启动，多机单机均可（本脚本按单机多卡典型使用）
- 仅 rank0 打印/写日志（tqdm、CSV、TensorBoard），保存 best/last 权重
"""

import torch.nn.functional as F
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, random_split, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import csv

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# 模型定义
class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, p=None, dropout=0.0):
        super().__init__()
        if p is None:
            p = d
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, dilation=d, padding=p, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class PedalModel4CNNAttn(nn.Module):
    def __init__(self, in_feats=128, hidden=96, heads=8, dropout=0.1,
                 use_pool=True, pool_ks=2, pool_stride=2):
        super().__init__()
        self.hidden = hidden
        self.use_pool = use_pool
        self.pool_ks = pool_ks
        self.pool_stride = pool_stride

        # ---- Conv stack: Conv-Conv-Pool × 2 ----
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_feats, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ks, stride=pool_stride) if use_pool else nn.Identity()

        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool_ks, stride=pool_stride) if use_pool else nn.Identity()

        self.proj = nn.Conv1d(hidden, hidden, kernel_size=1)

        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, 1) 
    def forward(self, x):
        import torch.nn.functional as F
        assert x.dim() == 3, f"expect 3D input, got {x.shape}"

        if x.shape[-1] == 128:
            T_orig = x.shape[1]
            x = x.transpose(1, 2).contiguous()           
        elif x.shape[1] == 128:
            T_orig = x.shape[2]
        else:
            raise RuntimeError(f"unexpected input shape for Conv1d: {x.shape} (need last or 2nd dim == 128)")

        if getattr(self, "_dbg_once", None) is None:
            #print(f"[FWD] normalized in: {x.shape}, T_orig={T_orig}")
            self._dbg_once = True

        # ===== 卷积栈 =====
        x = self.conv1(x)                                  
        x = self.conv2(x)                                 
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.proj(x)                                  

        x = x.transpose(1, 2).contiguous()                
        x = x.permute(1, 0, 2)                            

        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        x = self.out(x)                                   
        x = x.permute(1, 0, 2).contiguous()               
        x = x.squeeze(-1)                                 

        if getattr(self, "use_pool", False):
            x = x.unsqueeze(1)                             
            x = F.interpolate(x, size=T_orig, mode="linear", align_corners=False)
            x = x.squeeze(1)                               

        assert x.shape[1] == T_orig, f"pred T={x.shape[1]} != label T={T_orig}"
        return x

# DDP 辅助
def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# 训练入口
def main():
    ap = argparse.ArgumentParser(description="DDP training for CC64 prediction (4CNN+Attention)")
    # 数据/日志
    ap.add_argument("--npz", required=True, help="预处理数据 npz（包含 features[N,T,128], labels[N,T]）")
    ap.add_argument("--log_dir", default="runs/pedal_ddp", help="日志与权重路径")
    ap.add_argument("--csv_log", default="train_log.csv", help="CSV 日志文件名")
    ap.add_argument("--tensorboard", action="store_true", help="启用 TensorBoard")
    # 模型/训练
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--heads",  type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_per_gpu", type=int, default=512, help="每块 GPU 的 batch 大小")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=0.1, help="HuberLoss delta")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="按样本随机切分验证比例（简化版）")
    ap.add_argument("--amp", action="store_true", help="开启混合精度")
    ap.add_argument("--early_stop", type=int, default=8, help="patience，0=不启用")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="显著提升阈值")
    ap.add_argument("--print_every", type=int, default=1, help="tqdm 每多少步刷新后缀")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    set_seed(42)

    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 只在 rank0 创建目录/日志文件
    if is_main_process(rank):
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # ---------- 数据 ----------
    data = np.load(args.npz, allow_pickle=True)
    X = torch.tensor(data["features"], dtype=torch.float32) 
    Y = torch.tensor(data["labels"],   dtype=torch.float32)  
    # Conv1d 期望 [N,F,T]
    X = X.permute(0, 2, 1) 

    dataset = TensorDataset(X, Y)
    n_total = len(dataset)
    n_val   = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    train_loader  = DataLoader(train_ds, batch_size=args.batch_per_gpu, sampler=train_sampler,
                               num_workers=args.workers, pin_memory=True, persistent_workers=False)
    val_loader    = DataLoader(val_ds,   batch_size=args.batch_per_gpu, sampler=val_sampler,
                               num_workers=args.workers, pin_memory=True, persistent_workers=False)

    # 模型/优化
    model = PedalModel4CNNAttn(in_feats=X.shape[1], hidden=args.hidden, dropout=args.dropout, heads=args.heads).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.HuberLoss(delta=args.delta)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)) \
                if len(train_loader) > 0 else None

    scaler = torch.amp.GradScaler(enabled=args.amp)

    # 日志
    writer = None
    if args.tensorboard and SummaryWriter is not None and is_main_process(rank):
        writer = SummaryWriter(log_dir=args.log_dir)

    csv_path = os.path.join(args.log_dir, args.csv_log)
    if is_main_process(rank) and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "steps_per_sec", "elapsed_sec"])

    best_val = float("inf")
    no_improve = 0

    # 训练循环 
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        start_t = time.time()
        run_loss = 0.0
        steps = 0

        iterator = train_loader
        if is_main_process(rank):
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

        for i, (bx, by) in enumerate(iterator, start=1):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    pred = model(bx)
                    loss = criterion(pred, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(bx) 
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            run_loss += loss.item()
            steps += 1

            if is_main_process(rank) and (i % args.print_every == 0):
                avg = run_loss / steps
                elapsed = time.time() - start_t
                sps = steps / max(1e-6, elapsed)
                iterator.set_postfix(loss=f"{avg:.5f}", sps=f"{sps:.1f}")

        # 汇总 train_loss（各进程求平均）
        train_loss_local = torch.tensor([run_loss / max(1, steps)], dtype=torch.float32, device=device)
        dist.all_reduce(train_loss_local, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_local / world_size).item()

        # 验证 
        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                pred = model(bx)
                val_run += criterion(pred, by).item()

        # 汇总 val_loss
        val_loss_local = torch.tensor([val_run / max(1, len(val_loader))], dtype=torch.float32, device=device)
        dist.all_reduce(val_loss_local, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_local / world_size).item()

        elapsed = time.time() - start_t
        sps = steps / max(1e-6, elapsed)

        min_delta = getattr(args, "min_delta", 1e-4)
        stop_now = False

        if is_main_process(rank):
            print(f"[Epoch {epoch:02d}] train {train_loss:.5f} | val {val_loss:.5f} | {sps:.1f} steps/s | {elapsed/60:.1f} min")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{sps:.2f}", f"{elapsed:.2f}"])
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val",   val_loss,   epoch)
                writer.add_scalar("Speed/steps_per_sec", sps, epoch)

            # 保存权重
            torch.save(model.module.state_dict(), os.path.join(args.log_dir, "pedal_last.pth"))
            improved = (best_val - val_loss) > min_delta
            stop_now = False
            if improved:
                best_val = val_loss
                no_improve = 0
                torch.save(model.module.state_dict(), os.path.join(args.log_dir, "pedal_best.pth"))
                print("  ↳ new best saved: pedal_best.pth")
            else:
                no_improve += 1
                if args.early_stop > 0 and no_improve >= args.early_stop:
                    print(f"Early stopping at epoch {epoch} (patience {args.early_stop})")
                    stop_now = True

        # 所有进程同步早停状态
        # 只有 rank0 才可能将 stop_now 置 True，这里各 rank 合并（求和）后判断
        local_stop = torch.tensor([1 if (is_main_process(rank) and 'stop_now' in locals() and stop_now) else 0], device=device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(local_stop, op=dist.ReduceOp.SUM)

        if local_stop.item() > 0:
            if dist.is_initialized():
                dist.barrier()  
            break

    if writer is not None and is_main_process(rank):
        writer.close()

    if is_main_process(rank):
        print("✅ Training finished.")
        print(f"   Best val loss: {best_val:.6f}")
        print(f"   Logs & weights saved in: {args.log_dir}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
