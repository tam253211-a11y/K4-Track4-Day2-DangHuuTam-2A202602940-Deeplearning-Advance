"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None  (model.train() nhưng giữ BN đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
    weight_tag(model) -> str  (tag trọng số timm thực sự được tải, ghi vào results.xlsx)
"""
from __future__ import annotations

import torch
from torch import nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
INIT_CHOICES = ("scratch", "frozen", "finetune")
_NORM_TYPES = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm)


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp bằng timm (head mới, khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : không tải trọng số, huấn luyện toàn bộ
      - "frozen"   : trọng số tiền huấn luyện, đóng băng backbone, chỉ train head
      - "finetune" : trọng số tiền huấn luyện, train toàn bộ
    `init` quyết định việc tải trọng số; `pretrained=False` chỉ có tác dụng với "finetune"/"frozen" để debug.
    """
    import timm

    assert init in INIT_CHOICES, f"init phải thuộc {INIT_CHOICES}, nhận {init!r}"
    name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tên đầy đủ `kiến_trúc.tag` của trọng số đã tải (vd 'resnet50.a1_in1k'); 'scratch' nếu không tải."""
    if getattr(model, "init_mode", None) == "scratch":
        return "scratch"
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch, tag = cfg.get("architecture", type(model).__name__), cfg.get("tag")
    return f"{arch}.{tag}" if tag else arch


def data_config(model) -> dict:
    """mean/std/kích thước đầu vào mà trọng số timm mong đợi (dùng cho dataset.build_transforms)."""
    import timm

    return timm.data.resolve_data_config({}, model=model)


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của phần đóng băng phải ở chế độ eval, nếu không running_mean/var vẫn bị cập nhật
    bằng thống kê batch dù trọng số không đổi. model.train() sẽ bật lại BN, nên train loop phải gọi
    set_train_mode(model) thay cho model.train().
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True
    set_train_mode(model)


def set_train_mode(model) -> None:
    """model.train(), nhưng các module norm có tham số bị đóng băng (hoặc cả backbone đóng băng) giữ eval."""
    model.train()
    if not getattr(model, "frozen_backbone", False):
        return
    head_modules = set(model.get_classifier().modules())
    for m in model.modules():
        if m not in head_modules and isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    # dropout trước head vẫn train; phần backbone còn lại không có trạng thái phụ thuộc chế độ.


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1                 : lr = lr_backbone, weight_decay = weight_decay
    - norm, bias, token/pos-embed backbone : lr = lr_backbone, weight_decay = 0
      (ndim <= 1, cộng các tham số mà timm khai báo trong model.no_weight_decay(), vd cls_token, pos_embed)
    - head mới                             : lr = lr_head,     weight_decay = weight_decay
    Bỏ qua tham số requires_grad == False; bỏ nhóm rỗng.
    """
    head = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head:
            head_params.append(p)
        elif p.ndim <= 1 or name in skip or name.split(".")[-1] in skip:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"name": "backbone", "params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"name": "backbone_no_decay", "params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"name": "head", "params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC, không phải FLOPs 2x).

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch, không cần cài thêm).
    Nó đếm FLOPs của conv/matmul/attention (= 2 x MAC) nên chia 2; các phép từng phần tử
    (activation, norm, cộng residual) không được đếm, giống quy ước của fvcore/bảng timm.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    model.train(was_training)
    if was_training:
        set_train_mode(model)
    return counter.get_total_flops() / 2 / 1e9
