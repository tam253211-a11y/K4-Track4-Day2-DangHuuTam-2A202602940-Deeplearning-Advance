"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits[N, 9]] * K)
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import contextlib
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _amp_ctx(device, amp: bool):
    if amp and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def predict_views(model, loader, device, views, amp: bool = False, channels_last: bool = False):
    """Một lượt qua loader, chạy model trên K view của mỗi batch (đọc/giải mã ảnh chỉ một lần).

    `views` là list các hàm batch -> batch (hoặc batch -> list batch, được nối phẳng). Trả về
    (filenames, y_true, [logits_view_k (N, 9) cho k = 1..K]) theo đúng thứ tự của loader.
    """
    model.eval()
    dtype = next(model.parameters()).dtype          # model.half() -> đưa ảnh về fp16
    names, ys, outs = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True).to(dtype)
        batches = []
        for v in views:
            out = v(x)
            batches.extend(out if isinstance(out, (list, tuple)) else [out])
        if outs is None:
            outs = [[] for _ in batches]
        for k, xb in enumerate(batches):
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _amp_ctx(device, amp):
                outs[k].append(model(xb).float().cpu())
        names.extend(f)
        ys.append(torch.as_tensor(y))
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = False, channels_last: bool = False):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9]).
    """
    names, y, logits = predict_views(model, loader, device, [view or view_identity], amp, channels_last)
    return names, y, logits[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch lớn hơn (vd ảnh gốc 256 -> crop 224);
    flip=True thêm bản lật ngang của cả 5 (10 crop). Trả về list các batch."""
    h, w = x.shape[-2:]
    assert h >= crop and w >= crop, f"ảnh {h}x{w} nhỏ hơn crop {crop}"
    top, left = (h - crop) // 2, (w - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], x[..., top:top + crop, left:left + crop]]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias khi thu nhỏ), trả về list các batch.

    CNN có global pooling nhận được mọi kích thước. ViT/DeiT cần nội suy position embedding, Swin cần
    kích thước chia hết cho cửa sổ: notebook chỉ dò độ phân giải cho CNN và ghi rõ giới hạn này.
    """
    out = []
    for s in sizes:
        if x.shape[-1] == s and x.shape[-2] == s:
            out.append(x)
        else:
            out.append(F.interpolate(x.float(), size=(s, s), mode="bilinear", align_corners=False,
                                     antialias=s < x.shape[-1]).to(x.dtype))
    return out


def softmax_np(logits, axis: int = -1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá.
    """
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in logits_per_view])  # (K, N, C)
    if space == "prob":
        p = softmax_np(stack).mean(0)
    elif space == "logit":
        p = softmax_np(stack.mean(0))
    else:
        raise ValueError(f"space phải là 'prob' hoặc 'logit', nhận {space!r}")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file
    (người gọi kiểm tra thứ tự filename trước khi ghép).
    """
    shapes = {np.shape(p) for p in list_of_probs}
    assert len(shapes) == 1, f"các mô hình phải có cùng số ảnh và số lớp, nhận {shapes}"
    p = np.mean(np.stack(list_of_probs).astype(np.float64), axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu log T (để T luôn dương) bằng LBFGS trên float64, khởi đầu từ T tốt nhất của một lưới thô
    (tránh cực tiểu địa phương/không hội tụ). Accuracy không đổi vì thứ tự lớp không đổi.
    KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)

    def nll(t):
        return F.cross_entropy(z / t, y)

    grid = torch.logspace(-1, 1, 41, dtype=torch.float64)        # T trong [0,1; 10]
    t0 = grid[torch.stack([nll(t) for t in grid]).argmin()]
    log_t = torch.log(t0).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return t if nll(torch.tensor(t)) <= nll(t0) else float(t0)


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> nn.Conv2d:
    """Conv2d mới (có bias) = conv rồi BN ở chế độ eval:
    w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = copy.deepcopy(conv)
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float64,
                                                                               device=w.device)
    mean, var = bn.running_mean.double(), bn.running_var.double()
    gamma = bn.weight.detach().double() if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(mean)
    scale = gamma / torch.sqrt(var + bn.eps)
    fused.weight = nn.Parameter((w * scale.view(-1, 1, 1, 1)).to(conv.weight.dtype))
    fused.bias = nn.Parameter((beta + (b - mean) * scale).to(conv.weight.dtype))
    return fused


def _bn_replacement(bn) -> nn.Module:
    """BN thường -> Identity. timm BatchNormAct2d (BN + drop + activation trong một module, dùng ở
    EfficientNet/MobileNetV3) -> giữ lại phần drop + activation."""
    act, drop = getattr(bn, "act", None), getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop or nn.Identity(), act or nn.Identity())


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Trả về BẢN SAO đã gộp (model gốc không đổi), ở chế độ eval, với thuộc tính `n_fused_bn` = số cặp
    đã gộp. Chỉ gộp khi BN đứng NGAY sau một Conv2d trong cùng module cha (thứ tự đăng ký module,
    ví dụ conv1 -> bn1 của ResNet, Sequential(conv, bn) của nhánh downsample). Kiến trúc không có BN
    (ViT, Swin, ConvNeXt dùng LayerNorm) cho n_fused_bn = 0: không áp dụng. Kiểm tra sai số bằng
    `check_fusion`.
    """
    fused = copy.deepcopy(model).eval()
    n = 0

    def visit(parent):
        nonlocal n
        prev_name, prev = None, None
        for name, child in list(parent.named_children()):
            if (isinstance(child, nn.modules.batchnorm._BatchNorm) and isinstance(prev, nn.Conv2d)
                    and child.track_running_stats and prev.out_channels == child.num_features):
                setattr(parent, prev_name, _fuse_pair(prev, child))
                setattr(parent, name, _bn_replacement(child))
                n += 1
                prev_name, prev = None, None
                continue
            visit(child)
            prev_name, prev = name, child

    visit(fused)
    fused.n_fused_bn = n
    return fused


@torch.inference_mode()
def check_fusion(model, fused, x) -> float:
    """Sai số tuyệt đối lớn nhất giữa đầu ra model gốc và bản đã gộp BN (kỳ vọng ~1e-5 trở xuống)."""
    model.eval(); fused.eval()
    return float((model(x).float() - fused(x).float()).abs().max())
