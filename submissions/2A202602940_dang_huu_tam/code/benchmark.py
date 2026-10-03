"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý: chỉ đo forward của model trên tensor đã nằm sẵn trên GPU
"""
from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
import torch

DTYPES = ("fp32", "amp", "fp16")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (vd torch.cuda.synchronize) hoặc None trên CPU. Mỗi lần đo:
    sync(); t0; fn(); sync(); t1  -> thời gian GPU thật sự chạy xong, không chỉ thời gian xếp lệnh.
    """
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)), "mean": float(times.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, channels_last: bool = False,
                   bn_fused: bool = False) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (bản sao model.half(), model gốc không đổi).
    Trả về dict ghi thẳng vào sheet `Latency`. `bn_fused` chỉ để ghi nhãn (gộp BN làm ở inference.py).
    Lưu ý: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73) - đo thật, đừng giả định.
    """
    assert dtype in DTYPES, f"dtype phải thuộc {DTYPES}"
    dev = torch.device(device)
    m = copy.deepcopy(model).half() if dtype == "fp16" else model
    m = m.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        x = x.half()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    ctx = (torch.autocast(device_type=dev.type, dtype=torch.float16) if dtype == "amp"
           else contextlib.nullcontext())

    @torch.inference_mode()
    def fn():
        with ctx:
            m(x)

    r = bench(fn, warmup, iters, torch.cuda.synchronize if dev.type == "cuda" else None)
    return {"gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "channels_last": channels_last,
            "bn_fused": bn_fused, "preprocessing": False, **r,
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, img_size: int = 224, **kw) -> dict:
    """Độ trễ của TTA K view cho MỘT ảnh: K view xếp thành một batch K rồi forward một lần
    (đúng cách inference.py chạy TTA). So với 1 view và với K x p50(1 view) (slide trang 63)."""
    one = latency_report(model, 1, img_size, **kw)
    tta = latency_report(model, k_views, img_size, **kw)
    return {**tta, "batch": 1, "k_views": k_views, "images_per_s": 1000 / tta["p50"],
            "p50_1view": one["p50"], "k_times_p50_1view": k_views * one["p50"],
            "ratio_vs_1view": tta["p50"] / one["p50"]}
