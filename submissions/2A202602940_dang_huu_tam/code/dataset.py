"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

torch/torchvision chỉ import trong hàm, để load_split/check_split chạy được trên máy không có torch.
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Dataset chỉ là giao thức map-style; DataLoader vẫn dùng được nếu thiếu torch lúc import
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object

NUM_CLASSES = 9
TOTAL_IMAGES = 17_509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_CROP_PCT = 0.875                  # 224 / 256: resize cạnh về img_size/0.875 rồi center-crop img_size
AUG_CHOICES = ("basic", "vflip", "color", "trivial", "randaug")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    File fold của tác giả chỉ có cột `Filename, Label` (không có `Species`). Trả về ba DataFrame,
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        assert not missing, f"{split}_subset{fold}.csv thiếu cột {missing}"
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, total_expected: int = TOTAL_IMAGES) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Lỗi ở bất kỳ ý nào thì AssertionError để dừng ngay:
      1. số ảnh mỗi tập lệch 60/20/20 không quá 1 điểm phần trăm; in số ảnh mỗi lớp trong từng tập
      2. giao từng cặp tập theo Filename rỗng
      3. hợp ba tập đúng `total_expected` ảnh
      4. mọi Filename tồn tại trong `images_dir`
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    files = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert len(files[k]) == len(v), f"{k}: có Filename trùng lặp trong cùng một tập"

    n = {k: len(v) for k, v in splits.items()}
    total = sum(n.values())
    ratio = {k: n[k] / total for k in n}
    for k, target in {"train": 0.6, "val": 0.2, "test": 0.2}.items():
        assert abs(ratio[k] - target) <= 0.01, f"{k} chiếm {ratio[k]:.2%}, lệch 60/20/20 quá 1 điểm %"

    per_class = pd.DataFrame({k: v["Label"].value_counts().sort_index() for k, v in splits.items()})
    per_class = per_class.reindex(range(NUM_CLASSES)).fillna(0).astype(int)
    per_class["total"] = per_class.sum(axis=1)
    per_class.index = [f"{i} {CLASS_NAMES[i]}" for i in per_class.index]

    overlap = {
        "train&val": len(files["train"] & files["val"]),
        "train&test": len(files["train"] & files["test"]),
        "val&test": len(files["val"] & files["test"]),
    }
    assert not any(overlap.values()), f"Giao giữa các tập phải rỗng: {overlap}"

    union = files["train"] | files["val"] | files["test"]
    assert len(union) == total_expected, f"Hợp ba tập có {len(union)} ảnh, kỳ vọng {total_expected}"

    on_disk = set(os.listdir(images_dir))
    missing = sorted(union - on_disk)
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:5]}"

    print("Số ảnh:", {k: f"{n[k]} ({ratio[k]:.1%})" for k in n}, "| tổng", total)
    print("Giao từng cặp:", overlap, "| hợp:", len(union), "| thiếu file: 0")
    print(per_class.to_string())
    return {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap,
            "union": len(union), "missing_files": 0}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform.

    Train, theo `aug` (trục B của GUIDE.md mục 3; Mixup/CutMix trộn theo batch nên nằm ở losses.py):
      "basic"   : RandomResizedCrop(img_size) + lật ngang
      "vflip"   : basic + lật dọc (ảnh chụp thảm thực vật từ trên xuống, không có "phía trên" cố định)
      "color"   : basic + ColorJitter (ánh sáng ngoài đồng thay đổi theo giờ/mùa)
      "trivial" : basic + TrivialAugmentWide
      "randaug" : basic + RandAugment(num_ops=2, magnitude=9)
    Val/test: resize cạnh ngắn về round(img_size / 0.875) rồi CenterCrop(img_size). Với img_size=224 là
    resize 256 (đúng kích thước gốc) + CenterCrop 224. Không có augmentation ngẫu nhiên.
    """
    from torchvision import transforms as T

    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = round(img_size / EVAL_CROP_PCT)
        return T.Compose([T.Resize(resize), T.CenterCrop(img_size), *normalize])

    assert aug in AUG_CHOICES, f"aug phải thuộc {AUG_CHOICES}, nhận {aug!r}"
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "vflip":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose([*ops, *normalize])


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str); tên file dùng để ghi
    `predictions/*.csv` đúng định dạng của eval.py.

    cache=True nạp trước các byte JPEG vào RAM (toàn bộ dataset ~490 MB) để tránh nghẽn đọc đĩa;
    ảnh vẫn được giải mã ở mỗi lần đọc nên augmentation không bị ảnh hưởng.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self._bytes = None
        if cache:
            self._bytes = [(self.images_dir / f).read_bytes() for f in self.filenames]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        fname = self.filenames[i]
        src = io.BytesIO(self._bytes[i]) if self._bytes is not None else self.images_dir / fname
        with Image.open(src) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], fname


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random trong từng worker từ seed torch đã cấp cho worker (tái lập augmentation)."""
    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def class_balanced_weights(labels) -> np.ndarray:
    """Trọng số mỗi mẫu = 1 / (số ảnh của lớp đó), dùng cho WeightedRandomSampler."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return 1.0 / counts[labels]


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool = False, drop_last: bool | None = None):
    """Tạo DataLoader.

    - train=True: shuffle, hoặc sampler="balanced" (WeightedRandomSampler, trọng số 1/số ảnh của lớp,
      rút có hoàn lại đủ len(df) mẫu mỗi epoch; trục D của GUIDE.md mục 3).
    - train=False: không shuffle, giữ đúng thứ tự df để ghép logit với Filename.
    - drop_last mặc định True khi train (batch cuối quá nhỏ làm BatchNorm không ổn định).
    - `seed` cố định thứ tự batch/sampler; seed_worker cố định augmentation trong worker.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)

    shuffle, smp = False, None
    if train:
        if sampler == "balanced":
            w = torch.as_tensor(class_balanced_weights(ds.labels), dtype=torch.double)
            smp = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        elif sampler is None:
            shuffle = True
        else:
            raise ValueError(f"sampler phải là None hoặc 'balanced', nhận {sampler!r}")
    elif sampler is not None:
        raise ValueError("Không dùng sampler khi đánh giá: thứ tự phải giữ nguyên theo df")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=train if drop_last is None else drop_last,
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=num_workers > 0,
    )
