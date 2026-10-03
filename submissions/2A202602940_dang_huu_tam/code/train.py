"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm.

Mỗi lần chạy ghi vào run_dir = <out_dir>/<exp_id>/seed<k>/:
    config.json, history.csv, steps.csv (LR theo bước), best.pt, last.pt (để tiếp tục khi phiên bị ngắt),
    val_logits.npy, [test_logits.npy], summary.json
và ghi thêm <pred_dir>/<exp_id>_seed<k>_val.csv, [<exp_id>_seed<k>_test.csv], <curves_dir>/<exp_id>_<desc>.png.
Nếu summary.json đã có thì run() trả về luôn (chạy lại notebook không train lại); last.pt có thì tiếp tục.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import dataset as D
import losses as L
import model as M

_REPO_ROOT = Path(__file__).resolve().parents[3]  # submissions/<mssv>_<ten>/code/train.py -> gốc repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402  (eval.py gốc, không sửa)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png (mặc định = backbone)
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | vflip | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.1      # dùng khi loss = "ls"
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None  # ce_weighted: None -> 1/n_c; focal: None -> không alpha
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9             # chỉ dùng cho sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    cache_images: bool = True         # nạp sẵn byte JPEG vào RAM (~490 MB) để không nghẽn đọc đĩa
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """curves/<exp_id>_<desc>.png; seed khác 0 thêm _seed<k> để các seed chung kết không ghi đè nhau."""
    desc = cfg.desc or cfg.backbone
    suffix = f"_seed{cfg.seed}" if cfg.seed != 0 else ""
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). Worker DataLoader được seed trong dataset.make_loader.

    cudnn.benchmark = True để nhanh: kết quả lặp lại gần đúng chứ không bit-for-bit (thuật toán conv
    được chọn theo thời gian đo). Đây là mức tái lập được ghi trong báo cáo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD+momentum, trục E) với 3 nhóm tham số (model.param_groups)."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer phải là adamw hoặc sgd, nhận {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (slide trang 55).

    Hệ số nhân LR: bước s < W: (s + 1) / W;  sau đó 0.5 * (1 + cos(pi * (s - W) / (S - W))).
    warmup_epochs = 0 thì chỉ có cosine.
    """
    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao `self.module` để đánh giá. Buffer dạng số thực (running_mean/var của BatchNorm)
    cũng được lấy trung bình động như tham số, nên thống kê BN khớp với trọng số EMA;
    buffer số nguyên (num_batches_tracked) chép thẳng.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        ema_state = self.module.state_dict()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                ema_state[k].copy_(v)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def _autocast(cfg: Config, device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=cfg.amp and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, max_batches: int | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc" (NaN nếu Mixup/CutMix), "lrs": LR head theo bước}.

    set_train_mode giữ BN của backbone đóng băng ở eval. Với Mixup/CutMix, accuracy train không
    còn nghĩa bình thường nên không tính.
    """
    M.set_train_mode(model)
    total_loss, correct, n, lrs = 0.0, 0, 0, []
    for i, (x, y, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast(cfg, device):
            if cfg.mix:
                x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                logits = model(x)
                loss = L.mixed_loss(criterion, logits, targets)
            else:
                logits = model(x)
                loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lrs.append(optimizer.param_groups[-1]["lr"])  # nhóm cuối là head
        scheduler.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * len(y)
        n += len(y)
        if not cfg.mix:
            correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": total_loss / n, "train_acc": correct / n if not cfg.mix else float("nan"),
            "lrs": lrs}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, cfg: Config | None = None):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float),
    giữ đúng thứ tự của loader. `criterion` ở đây nên là CE thường để loss val so sánh được giữa các cấu hình.
    """
    model.eval()
    names, ys, outs, total_loss = [], [], [], 0.0
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg is not None and cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=bool(cfg and cfg.amp and device.type == "cuda")):
            logits = model(x)
        logits = logits.float()
        total_loss += criterion(logits, y).item() * len(y)
        names.extend(f)
        ys.append(y.cpu())
        outs.append(logits.cpu())
    y_true = torch.cat(ys).numpy()
    return names, y_true, torch.cat(outs).numpy(), total_loss / len(y_true)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<desc>.png (GUIDE.md mục 6.2).

    Ba ô: loss train/val theo epoch; macro-F1 val (và top-1 val, accuracy train nếu có) theo epoch;
    LR của head theo bước (thấy warmup + cosine). Epoch tốt nhất được đánh dấu.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax()]
    fig, ax = plt.subplots(1, 3 if lrs else 2, figsize=(15 if lrs else 10, 4))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train loss")
    ax[0].plot(h.epoch, h.val_loss, "o-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h.epoch, h.val_macro_f1, "o-", label="val macro-F1")
    ax[1].plot(h.epoch, h.val_top1, "s--", label="val top-1")
    if h.train_acc.notna().any():
        ax[1].plot(h.epoch, h.train_acc, "^:", label="train acc")
    ax[1].axvline(best.epoch, color="gray", ls=":", label=f"best ep {int(best.epoch)}: F1 {best.val_macro_f1:.4f}")
    ax[1].set(xlabel="epoch", ylabel="score", title="Val metrics")
    if lrs:
        ax[2].plot(np.arange(len(lrs)), lrs)
        ax[2].set(xlabel="step", ylabel="LR (head)", title="LR schedule")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _build_criterion(cfg: Config, train_df: pd.DataFrame, device):
    counts = np.bincount(train_df["Label"], minlength=D.NUM_CLASSES)  # chỉ dùng số liệu TRAIN
    if cfg.loss == "ce_weighted":
        w = L.class_weights(counts, cfg.class_weight_beta or 0.0)
        return L.build_criterion("ce_weighted", weight=w).to(device)
    if cfg.loss == "focal":
        alpha = None if cfg.class_weight_beta is None else L.class_weights(counts, cfg.class_weight_beta)
        return L.build_criterion("focal", gamma=cfg.focal_gamma, alpha=alpha).to(device)
    return L.build_criterion(cfg.loss, smoothing=cfg.label_smoothing).to(device)


def _setup(cfg: Config, device, need_test: bool = False):
    """Dựng split (đã kiểm tra S1-S6), model, transform đúng mean/std của trọng số, các loader."""
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir)
    model = M.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = M.data_config(model)
    mean, std = dc["mean"], dc["std"]
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    tf_eval = D.build_transforms(False, cfg.img_size, mean=mean, std=std)
    kw = dict(num_workers=cfg.num_workers, seed=cfg.seed, cache=cfg.cache_images)
    loaders = {
        "train": D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler, **kw),
        "val": D.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size * 2, False, **kw),
    }
    if need_test:
        loaders["test"] = D.make_loader(test_df, cfg.images_dir, tf_eval, cfg.batch_size * 2, False, **kw)
    return model, loaders, train_df, {"mean": list(mean), "std": list(std)}


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt.

    Thứ tự: seed + config.json -> split (check_split) -> loader -> model/criterion/optimizer/scheduler/
    scaler/EMA -> mỗi epoch: train, evaluate(val), ghi history, lưu best.pt theo MACRO-F1 VAL
    (hòa thì giữ epoch sớm hơn) và last.pt -> nạp best -> val logits + predictions val
    -> [test đúng MỘT lần nếu cfg.save_test_predictions] -> history.csv, curves, summary.json.
    KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    rd = run_dir(cfg)
    if (rd / "summary.json").exists():
        summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong trước đó, bỏ qua (xoá {rd} để chạy lại)")
        if cfg.save_test_predictions and not pred_path(cfg, "test").exists():
            summary["test_predictions"] = str(predict_test(cfg))
            (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, loaders, train_df, norm = _setup(cfg, device, need_test=cfg.save_test_predictions)
    criterion = _build_criterion(cfg, train_df, device)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    meta = {
        "config": dataclasses.asdict(cfg), "weight_tag": M.weight_tag(model),
        "params_M": M.count_params(model), "gmacs": M.count_gmacs(model, cfg.img_size),
        "normalize": norm, "steps_per_epoch": steps_per_epoch,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "timm": __import__("timm").__version__,
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }
    (rd / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    history, lrs, best_f1, best_epoch, start_epoch = [], [], -1.0, -1, 1
    if (rd / "last.pt").exists():  # phiên trước bị ngắt: tiếp tục từ epoch kế tiếp
        ck = torch.load(rd / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        history, lrs, best_f1, best_epoch = ck["history"], ck["lrs"], ck["best_f1"], ck["best_epoch"]
        start_epoch = ck["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start_epoch}")

    eval_model = ema.module if ema is not None else model
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0
        names, y_val, logits, val_loss = evaluate(eval_model, loaders["val"], eval_criterion, device, cfg)
        probs = softmax(logits)
        m = compute_metrics(y_val, probs.argmax(1), probs)
        lrs += tr.pop("lrs")
        row = {"epoch": epoch, **tr, "val_loss": val_loss, "val_macro_f1": m["macro_f1"],
               "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"],
               "train_time_s": train_time, "epoch_time_s": time.perf_counter() - t0}
        history.append(row)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch:2d}/{cfg.epochs} | train {tr['train_loss']:.4f} "
              f"| val {val_loss:.4f} | F1 {m['macro_f1']:.4f} | top1 {m['top1']:.4f} | {train_time:.0f}s")
        if m["macro_f1"] > best_f1:  # dấu > nghiêm ngặt: hòa thì giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), rd / "best.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "history": history, "lrs": lrs,
                    "best_f1": best_f1, "best_epoch": best_epoch}, rd / "last.pt")

    eval_model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=True))
    names, y_val, logits, _ = evaluate(eval_model, loaders["val"], eval_criterion, device, cfg)
    np.save(rd / "val_logits.npy", logits)
    probs = softmax(logits)
    save_predictions(pred_path(cfg, "val"), names, y_val, probs)
    val_m = compute_metrics(y_val, probs.argmax(1), probs)

    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
               "weight_tag": meta["weight_tag"], "params_M": meta["params_M"], "gmacs": meta["gmacs"],
               "best_epoch": best_epoch, "val_macro_f1": val_m["macro_f1"], "val_top1": val_m["top1"],
               "val_balanced_acc": val_m["balanced_acc"], "val_ece": val_m["ece"],
               "val_f1_per_class": val_m["f1"].tolist(),
               "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
               "gpu": meta["versions"]["gpu"]}

    if cfg.save_test_predictions:  # Bước 4: test đúng MỘT lần, bằng checkpoint đã chọn trên val
        names_t, y_test, logits_t, _ = evaluate(eval_model, loaders["test"], eval_criterion, device, cfg)
        np.save(rd / "test_logits.npy", logits_t)
        save_predictions(pred_path(cfg, "test"), names_t, y_test, softmax(logits_t))
        summary["test_predictions"] = str(pred_path(cfg, "test"))

    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    pd.DataFrame({"step": np.arange(len(lrs)), "lr_head": lrs}).to_csv(rd / "steps.csv", index=False)
    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}", lrs)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd / "last.pt").unlink(missing_ok=True)  # đã xong; best.pt đủ cho Bước 3, bớt dung lượng Drive
    print(f"[{cfg.exp_id} seed{cfg.seed}] best epoch {best_epoch} | val macro-F1 {val_m['macro_f1']:.4f} "
          f"| top1 {val_m['top1']:.4f}")
    return summary


def predict_test(cfg: Config) -> Path:
    """Bước 4 cho một lần chạy ĐÃ train xong (vd mốc T00 seed0 từ Bước 2): nạp best.pt (chọn trên val)
    và chạy test đúng MỘT lần. Không huấn luyện lại, không đổi checkpoint."""
    out = pred_path(cfg, "test")
    assert not out.exists(), f"{out} đã có: test chỉ chạy một lần mỗi seed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaders, _, _ = _setup(cfg, device, need_test=True)
    model.load_state_dict(torch.load(run_dir(cfg) / "best.pt", map_location=device, weights_only=True))
    names, y_test, logits, _ = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device, cfg)
    np.save(run_dir(cfg) / "test_logits.npy", logits)
    return save_predictions(out, names, y_test, softmax(logits))


def measure_epoch_time(cfg: Config, max_batches: int | None = None) -> dict:
    """Đo thời gian 1 epoch train + 1 lượt val cho lập ngân sách GPU (không ghi gì ra đĩa).

    max_batches giới hạn số batch train rồi ngoại suy ra cả epoch (bỏ 3 batch đầu khi ngoại suy
    vì còn khởi động cuDNN/worker). Mặc định chạy đủ 1 epoch.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaders, train_df, _ = _setup(cfg, device)
    criterion = _build_criterion(cfg, train_df, device)
    optimizer = build_optimizer(model, cfg)
    n_batches = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, n_batches)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    if max_batches is None:
        sync(); t0 = time.perf_counter()
        train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, cfg, device)
        sync(); train_s = time.perf_counter() - t0
    else:
        warm = 3
        train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, cfg, device,
                        max_batches=warm)
        sync(); t0 = time.perf_counter()
        train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, cfg, device,
                        max_batches=max_batches)
        sync(); train_s = (time.perf_counter() - t0) / max_batches * n_batches
    t0 = time.perf_counter()
    evaluate(model, loaders["val"], nn.CrossEntropyLoss(), device, cfg)
    sync(); val_s = time.perf_counter() - t0
    out = {"backbone": cfg.backbone, "batch_size": cfg.batch_size, "img_size": cfg.img_size,
           "train_epoch_s": train_s, "val_s": val_s, "epoch_s": train_s + val_s,
           "extrapolated": max_batches is not None,
           "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0}
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return out


def _parse_value(raw: str, annotation: str, default):
    if raw.lower() in ("none", "null"):
        if "None" not in annotation:
            raise ValueError(f"giá trị None không hợp lệ cho kiểu {annotation}")
        return None
    if "bool" in annotation:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"bool cần true/false, nhận {raw!r}")
        return raw.lower() in ("true", "1")
    if "int" in annotation:
        return int(raw)
    if "float" in annotation:
        return float(raw)
    return raw


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, raw = pair.split("=", 1)
        if key not in fields:
            raise KeyError(f"Config không có trường '{key}'. Các trường: {', '.join(fields)}")
        f = fields[key]
        out[key] = _parse_value(raw, str(f.type), f.default)
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè trường của Config")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
