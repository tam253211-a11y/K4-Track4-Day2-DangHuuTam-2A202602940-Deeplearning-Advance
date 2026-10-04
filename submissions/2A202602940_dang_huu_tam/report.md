# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Sinh viên:** Đặng Hữu Tâm · MSSV 2A202602940
**Repo:** https://github.com/tam253211-a11y/K4-Track4-Day2-DangHuuTam-2A202602940-Deeplearning-Advance
**Notebook (có output của lần chạy thật):** [`code/lab_day2.ipynb`](code/lab_day2.ipynb)

Mọi con số dưới đây lấy từ lần chạy trọn notebook trên **một máy RTX 4090** (mục 2.4), truy ngược được qua `results.xlsx`, `eval_out/` và `runs_logs/<exp_id>__seed<k>__{config,summary,history}`. Số test do `eval.py` tính từ `predictions/`. Số của bài báo gốc chỉ được **trích dẫn** để đối chiếu.

---

## 1. Tóm tắt

Phân loại 9 lớp DeepWeeds (fold 0 của tác giả), chỉ số chính macro-F1. Đã làm: 7 backbone với cùng công thức nền (Bước 1); 6 ablation một-yếu-tố trên 4 trục cộng 1 kết hợp, có đo nhiễu bằng 3 seed (Bước 2); 16 cấu hình suy luận kèm độ trễ p50/p95/p99 (Bước 3); chung kết 3 seed, test chạy một lần mỗi seed (Bước 4).
**Cấu hình tốt nhất (chọn hoàn toàn trên val):** `convnext_tiny.in12k_ft_in1k` + TrivialAugment (`T05`) + suy luận ở độ phân giải 256 (`I04_256`) + temperature scaling khớp trên val. **Test (3 seed): macro-F1 0,9774 ± 0,0016, top-1 0,9820 ± 0,0003, ECE 0,0050 ± 0,0026; recall Chinee apple 0,957, Snake weed 0,959.** Mốc `T00`+`I00`: macro-F1 0,9717 ± 0,0028 → Δ = +0,0057, lớn hơn std (0,0028). p95 batch 1 = 4,0 ms trên RTX 4090.
**Kết luận chính:** backbone quyết định phần lớn (macro-F1 val 0,69 → 0,97 giữa các kiến trúc), suy luận ở độ phân giải cao hơn đóng góp phần cải thiện rõ nhất còn lại, còn công thức huấn luyện chỉ tăng nhỏ: trên val vượt nhiễu, trên test thì không phân biệt được.

---

## 2. Dữ liệu và thiết lập

### 2.1 Dữ liệu và kiểm tra chia tập

- DeepWeeds (Olsen et al., 2019): 17.509 ảnh RGB 256×256, 9 lớp (8 loài cỏ dại + `Negatives`). Ảnh tải từ Zenodo, **MD5 `b7b30f96d466fba86016aa5a26606e0f` khớp**.
- **Fold 0** của tác giả (`train/val/test_subset0.csv`), tải nguyên bản, không sửa (quy tắc S1–S6). Val dùng cho mọi lựa chọn; test chỉ chạy ở Bước 4, một lần mỗi seed.

| Kiểm tra (`dataset.check_split`) | Kết quả |
|---|---|
| Số ảnh train / val / test | 10.501 / 3.501 / 3.507 (60,0 / 20,0 / 20,0 %) |
| train∩val, train∩test, val∩test | 0 / 0 / 0 |
| Hợp ba tập | 17.509 |
| File trong CSV thiếu trong thư mục ảnh | 0 |

| Lớp | Train | Val | Test | Tổng (đếm) | Bài báo (Table 1) | Chênh |
|---|---:|---:|---:|---:|---:|---:|
| 0 Chinee Apple | 675 | 225 | 226 | 1.126 | 1.125 | +1 |
| 1 Lantana | 637 | 213 | 213 | 1.063 | 1.064 | −1 |
| 2 Parkinsonia | 618 | 206 | 207 | 1.031 | 1.031 | 0 |
| 3 Parthenium | 613 | 204 | 205 | 1.022 | 1.022 | 0 |
| 4 Prickly Acacia | 637 | 212 | 213 | 1.062 | 1.062 | 0 |
| 5 Rubber Vine | 605 | 202 | 202 | 1.009 | 1.009 | 0 |
| 6 Siam Weed | 644 | 215 | 215 | 1.074 | 1.074 | 0 |
| 7 Snake Weed | 609 | 203 | 204 | 1.016 | 1.016 | 0 |
| 8 Negatives | 5.463 | 1.821 | 1.822 | 9.106 | 9.106 | 0 |

**EDA** (`eval_out/eda_class_distribution.png`, `eval_out/eda_samples.png`):

- Tổng khớp bài báo; Chinee Apple +1 và Lantana −1 so với Table 1 (có lẽ một ảnh đổi nhãn giữa CSV hiện hành và bảng in), tổng không đổi. Dùng nguyên CSV.
- **Mất cân bằng:** lớp lớn nhất / nhỏ nhất = 9.106 / 1.009 ≈ **9,0 lần**; `Negatives` chiếm **52,0%**, nên đoán toàn `Negatives` đã được ~52% top-1. Vì vậy macro-F1 là chỉ số chính. Tỉ lệ lớp gần như giống hệt giữa ba tập.
- 500 ảnh train ngẫu nhiên đều 256×256 RGB. Mean RGB ≈ (0,379; 0,393; 0,385), tối hơn ImageNet (0,485; 0,456; 0,406); vẫn dùng mean/std của trọng số timm vì tinh chỉnh từ ImageNet.
- **Nhìn ảnh mẫu:** ảnh chụp từ trên xuống, nền đất, cỏ khô, lá rụng chiếm phần lớn khung; ánh sáng rất khác nhau (bóng đổ đậm, cháy sáng, ám tím/hồng do cân bằng trắng). Chinee apple và Snake weed đều là cây lá xanh đậm hình bầu dục, mọc dày, khó tách bằng mắt khi ảnh tối. `Negatives` là thảm thực vật bản địa nhìn rất giống các loài cỏ dại, nên nhầm "loài X ↔ Negatives" là điều dễ đoán trước (mục 6.3 xác nhận).

### 2.1b Kiểm tra pipeline trước khi chạy thật

| # | Kiểm tra | Kết quả (output notebook) |
|---|---|---|
| 1 | Cố định seed (random, numpy, torch, worker) | Seed 0 lặp lại: cùng thứ tự file, cùng ảnh sau augmentation, cùng head; seed 1: khác |
| 2 | Loss CE ban đầu ≈ ln 9 = 2,197 | 2,203 (`model.py`), 2,226 (batch val thật, resnet50) |
| 3 | Overfit 18 ảnh (2 ảnh/lớp, 100 bước) | Loss 2,179 → 0,0020; accuracy (eval mode) 1,00 |
| 4 | Ảnh sau augmentation khớp nhãn | Nhãn 16 ảnh khớp CSV; ảnh ở `eval_out/check_augmentation.png` (kèm CutMix) |
| 5 | `train()`/`eval()` đúng lúc | Sau `evaluate()` mọi module ở eval; eval cho 2 lần forward giống nhau, train (dropout) thì khác |

Kiểm tra cài đặt tự viết (ô `losses.py`, `model.py`): focal γ=0 và label smoothing ε=0 bằng CE (sai số < 1e-6); CutMix qua 50 lần thử có `lam` đúng bằng diện tích ảnh gốc còn lại (kể cả khi hộp bị cắt ở biên) và vùng dán lấy từ đúng ảnh `y_b`; Mixup và `mixed_loss` đúng; đóng băng backbone chỉ còn head được train, 0 lớp BN ở train mode; trọng số lớp chỉ đếm trên train. Gộp BN (`inference.fuse_conv_bn`) kiểm tra trên CPU với thống kê BN ngẫu nhiên: sai số lớn nhất 7,5e-8 (resnet50), 7,2e-7 (efficientnet_b0), 1,2e-6 (mobilenetv3), 6,0e-8 (resnext50); convnext/deit không có BN nên gộp 0 lớp. Bộ test của repo: **38 test OK**.

### 2.2 Chỉ số

Chỉ số chính **macro-F1** (9 lớp, trọng số bằng nhau); phụ: top-1, balanced accuracy, precision/recall/F1 theo lớp, ECE 15 bin, NLL; độ trễ p50/p95/p99. mean ± std qua 3 seed, `ddof=1`. Số test tính bằng `eval.py` (không sửa).

### 2.3 Công thức nền `T00`

| Thành phần | Giá trị |
|---|---|
| Khởi tạo | ImageNet (timm), head mới 9 lớp, tinh chỉnh toàn bộ |
| Đầu vào | Train: `RandomResizedCrop(224)` + lật ngang. Val/test: resize 256 → `CenterCrop(224)` |
| Chuẩn hoá | mean/std theo cấu hình trọng số timm |
| Optimizer, LR | AdamW; backbone 1e-4, head 1e-3 |
| Weight decay | 0,05; không áp dụng cho norm, bias, token/pos-embed |
| Lịch LR | warmup tuyến tính 1 epoch + cosine về 0, cập nhật theo bước |
| Loss / batch / epoch | Cross-entropy / 64 / 12 (giống nhau cho mọi cấu hình) |
| Khác | AMP, `channels_last`; checkpoint = epoch có macro-F1 val cao nhất (hòa lấy epoch sớm hơn) |

### 2.4 Phần cứng, phần mềm, seed

- **Toàn bộ kết quả trong báo cáo: 1 GPU NVIDIA GeForce RTX 4090** (máy thuê vast.ai, 54 nhân CPU, `NUM_WORKERS = 12`), Python 3.12.14, PyTorch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.30. Notebook chạy trọn bằng `jupyter nbconvert --execute` (tổng thời gian chạy ~78 phút, kể cả một lần dừng do lỗi, mục 2.5).
- Seed: Bước 1–3 dùng seed 0; chung kết `F01` và mốc `T00` dùng seed 0, 1, 2. `cudnn.benchmark = True` nên lặp lại gần đúng, không trùng từng bit.
- **Lần chạy thăm dò trước đó trên Colab T4** (Bước 0, Bước 1, `T00` 3 seed) **không dùng trong bảng kết quả** để mọi số liệu cùng một phần cứng. Chúng cho thấy mức tái lập: macro-F1 val chênh ≤ 0,03 giữa hai lần chạy cùng cấu hình (ví dụ resnet50 0,7904 trên T4 so với 0,7933 trên 4090; convnext_tiny 0,9702 so với 0,9699), thứ hạng backbone giữ nguyên.

### 2.5 Ngân sách GPU, sự cố và các cắt giảm

- Đo 1 epoch trên Colab T4 (công thức nền, batch 64, AMP; đo đủ epoch, có `cuda.synchronize`): resnet50 49,5 s, resnext50 54,8 s, convnext_tiny 64,0 s, deit_small 51,7 s, swin_tiny 81,6 s, efficientnet_b0 48,4 s, mobilenetv3 45,2 s (train + val). Hai lần đo trên hai phiên Colab chênh tới 40% (efficientnet_b0: 68,7 s và 41,1 s), nên các số này chỉ dùng để lập kế hoạch. Kế hoạch đầy đủ ≈ 6 giờ T4 vượt hạn mức GPU miễn phí (Colab bị ngắt, Kaggle phải train lại từ đầu), nên lần chạy chính thức chuyển sang một RTX 4090 thuê (mỗi epoch convnext_tiny ~9 s).
- **GMAC không dự đoán được thời gian train:** trên T4, efficientnet_b0 (0,38 GMAC) và mobilenetv3 (0,22 GMAC) train một epoch không nhanh hơn resnet50 (4,09 GMAC) bao nhiêu. Trên 4090, `eval_out/epoch_timing.csv` có số lớn bất thường (ví dụ efficientnet 90 s) do lần đo đó gộp cả việc nạp ảnh vào RAM của loader mới; thời gian train/epoch đáng tin trên 4090 là cột trong sheet `Backbones` (đo trong lúc train, 6,6–11,2 s).
- **Sự cố:** lần chạy 4090 đầu tiên dừng ở Bước 2a với `OSError: [Errno 24] Too many open files` (12 worker × nhiều loader `persistent_workers` không được đóng giữa các lần train, vượt giới hạn 1.024 file). Đã sửa `train.py` (đóng worker sau mỗi lần chạy, chia sẻ tensor qua file, nâng giới hạn mềm) rồi chạy lại; các lần train đã xong được giữ nguyên (`run()` bỏ qua), `T02` chưa lưu epoch nào nên train lại từ đầu. Trên máy thuê, lệnh `python` không có trong PATH nên hai ô gọi shell (bộ test, `eval.py score/grade`) báo `command not found`; hai ô này không dùng GPU, đã sửa sang `sys.executable` và chạy lại trên máy cá nhân từ chính các file `predictions/`, output được ghi vào notebook.
- **Cắt giảm (theo thứ tự ưu tiên của GUIDE mục 7):**
  1. Ablation Bước 2 chỉ trên **1 backbone** (convnext_tiny). Kết luận về công thức chỉ chắc cho backbone này.
  2. Bước 2 rút còn **6 ablation trên 4 trục** (A: đóng băng; B: CutMix, TrivialAugment; C: label smoothing, focal; F: EMA) thay vì 12. Bỏ: train từ đầu (A), lật dọc, ColorJitter (B), CE có trọng số lớp (C), sampler cân bằng (D), cùng LR backbone/head (E). Các câu hỏi của trục D, E và "train từ đầu có kịp không" chưa được trả lời.
  3. Ablation chạy **1 seed**; 3 seed dành cho `T00` (đo nhiễu) và chung kết.
  4. 12 epoch cho mọi cấu hình (trong khoảng 10–15 của đề).

---

## 3. Kết quả so sánh backbone (Bước 1)

Bảy backbone, cùng công thức `T00`, cùng seed 0, cùng split (sheet `Backbones`). Độ trễ ở đây là đo sơ bộ (batch 1, FP32, 50 lần sau 10 lần warmup, có `cuda.synchronize`, không tính tiền xử lý).

| exp_id | Backbone (tag timm) | Params (M) | GMAC | macro-F1 val | top-1 val | F1 Chinee | F1 Snake | F1 val ep 1 | Best ep | Train/ep (s) | p50 / p95 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B03** | **convnext_tiny.in12k_ft_in1k** | 27,83 | 4,45 | **0,9699** | **0,9763** | 0,9519 | 0,9279 | 0,8441 | 10 | 8,6 | 4,00 / 4,09 |
| B04 | deit_small_patch16_224.fb_in1k | 21,67 | 4,60 | 0,9546 | 0,9663 | 0,9023 | 0,8862 | 0,8383 | 11 | 6,6 | 3,43 / 3,46 |
| B05 | swin_tiny_patch4_window7_224.ms_in1k | 27,53 | 4,49 | 0,9541 | 0,9669 | 0,9112 | 0,9073 | 0,7842 | 12 | 11,2 | 6,92 / 7,12 |
| B01 | resnet50.a1_in1k | 23,53 | 4,09 | 0,7933 | 0,8509 | 0,6199 | 0,6935 | 0,2101 | 11 | 7,2 | 4,51 / 4,56 |
| B06 | efficientnet_b0.ra_in1k | 4,02 | 0,38 | 0,7720 | 0,8323 | 0,6883 | 0,7174 | 0,3860 | 12 | 8,3 | 5,87 / 6,19 |
| B02 | resnext50_32x4d.a1h_in1k | 23,00 | 4,23 | 0,7635 | 0,8163 | 0,7379 | 0,7450 | 0,1886 | 11 | 8,3 | 4,44 / 4,50 |
| B07 | mobilenetv3_large_100.ra_in1k | 4,21 | 0,22 | 0,6887 | 0,7644 | 0,6931 | 0,6946 | 0,3653 | 6 | 6,6 | 4,61 / 4,71 |

Params đếm với head 9 lớp (nhỏ hơn số slide, vốn tính head 1000 lớp). Biểu đồ đánh đổi: `eval_out/backbones_tradeoff.png`; đường cong: `curves/B0x_<ten>.png`.

**Mức nhiễu.** Mỗi backbone 1 seed. `T00` (convnext_tiny) chạy 3 seed: 0,9699 / 0,9663 / 0,9707 → **0,9690 ± 0,0023**. Std này đo trên convnext_tiny, chỉ dùng làm cỡ độ lớn của nhiễu cho các backbone khác.

**Nhận xét.**

1. **Hai nhóm tách biệt rất rõ.** Ba mạng dùng LayerNorm (convnext_tiny, deit_small, swin_tiny): macro-F1 0,954–0,970. Bốn mạng dùng BatchNorm (resnet50, resnext50, efficientnet_b0, mobilenetv3): 0,69–0,79. Khoảng cách 0,16–0,28 lớn hơn nhiễu hàng chục lần.
2. **Trong nhóm đầu**, convnext_tiny hơn deit_small 0,015 và swin_tiny 0,016 (≈ 7 std), nên convnext_tiny tốt nhất là kết luận đáng tin dù mới 1 seed. deit_small và swin_tiny chỉ cách nhau 0,0005 < std: **không phân biệt được**.
3. **Hội tụ.** Nhóm LayerNorm đạt F1 0,78–0,84 ngay epoch 1 và đạt 98% F1 tốt nhất ở epoch 6–8. Nhóm BatchNorm chỉ 0,19–0,39 ở epoch 1; resnet50, resnext50, efficientnet_b0 vẫn tăng tới epoch 11–12 (chưa hội tụ trong 12 epoch), còn mobilenetv3 đạt đỉnh ở epoch 6 rồi quá khớp.
4. **Quá khớp.** mobilenetv3 rõ nhất: val loss thấp nhất ở epoch 6 rồi tăng tới 0,694 ở epoch 12 trong khi train loss giảm còn 0,22, F1 tốt nhất cũng ở epoch 6. resnext50 và efficientnet_b0 có khoảng cách train/val loss lớn (0,22 so với 0,53). Nhóm LayerNorm có val loss vẫn giảm tới epoch 12, không quá khớp rõ.
5. **Giả thuyết cho nhóm BatchNorm (chưa kiểm chứng).** Đặc trưng tiền huấn luyện của các mạng này không kém: linear probe nhanh (đặc trưng đóng băng + hồi quy logistic, 360 ảnh train / 225 ảnh val, CPU) cho macro-F1 0,68 (resnet50.a1), 0,74 (efficientnet_b0), so với 0,76 (convnext_tiny). Vấn đề nằm ở quá trình tinh chỉnh. Giả thuyết: `RandomResizedCrop(224)` với tỉ lệ 8–100% làm thống kê BatchNorm học trên ảnh phóng to, khác xa ảnh val nguyên khung, nên ở eval mode mô hình lệch; LayerNorm không lưu thống kê nên không bị. Kiểm chứng được bằng cách tính lại thống kê BN trên ảnh train với transform của val (không train lại); chưa làm. Đây cũng là câu trả lời một phần cho câu hỏi "chênh lệch ResNet ↔ ConvNeXt do kiến trúc hay công thức": với **cùng một công thức**, kiến trúc dùng BatchNorm chịu thiệt nặng.
6. **So với ImageNet.** Trên ImageNet các mạng cỡ ~25M tham số này chỉ cách nhau vài điểm top-1 (slide trang 41–42), còn ở đây convnext_tiny hơn resnet50 tới 0,18 macro-F1: thứ hạng ImageNet không dự đoán được thứ hạng trên DeepWeeds với công thức này. Lưu ý thêm `convnext_tiny.in12k_ft_in1k` được tiền huấn luyện trên ImageNet-12k, lớn hơn ImageNet-1k của các mạng còn lại.
7. **FLOPs không dự đoán được độ trễ batch 1** (sheet `Latency`, FP32): efficientnet_b0 (0,38 GMAC) 5,80 ms và mobilenetv3 (0,22 GMAC) 4,98 ms, **chậm hơn** resnet50 (4,09 GMAC) 4,45 ms và deit_small (4,60 GMAC) 3,41 ms; swin_tiny chậm nhất 6,88 ms. Ở batch 1 GPU không được dùng hết, thời gian bị chi phối bởi số lần gọi kernel. **Ở batch 32 thì FLOPs có ý nghĩa:** mobilenetv3 6.713 ảnh/s và efficientnet_b0 5.436 ảnh/s, so với resnet50 2.756 và convnext_tiny 2.185 ảnh/s.

**Chọn backbone cho Bước 2–3: `convnext_tiny` (B03).** macro-F1 val cao nhất (0,9699, hơn backbone thứ hai 0,015 ≈ 7 std), F1 hai lớp khó cao nhất (0,952 / 0,928), độ trễ batch 1 thuộc nhóm nhanh (p50 4,0 ms, nhanh hơn swin_tiny ~40%). deit_small nhanh hơn (3,4 ms) nhưng kém 0,015 macro-F1; với ngân sách 30–100 ms/khung, độ trễ không phải ràng buộc nên ưu tiên độ chính xác.

---

## 4. Kết quả công thức huấn luyện (Bước 2)

Trên convnext_tiny, mỗi `T0x` khác `T00` **đúng một yếu tố**, cùng seed 0, cùng 12 epoch. **Không dùng cách tham lam theo trục:** mọi `T0x` so với cùng nền `T00`; sau đó `T13` kết hợp giá trị tốt nhất của mỗi trục B–F (chọn tự động trên val). Ngưỡng nhiễu: std macro-F1 val của `T00` qua 3 seed = **0,0023** (sheet `Training`, biểu đồ `eval_out/training_ablation.png`).

| exp_id | Trục | Khác `T00` ở điểm nào | macro-F1 val | Δ so với T00 seed 0 | Δ / std | Kết luận | F1 Chinee | F1 Snake | F1 Negatives |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| T00 | – | nền, seed 0 / 1 / 2 | 0,9699 / 0,9663 / 0,9707 | – | – | 0,9690 ± 0,0023 | 0,9519 | 0,9279 | 0,9841 |
| T02 | A | đóng băng backbone, chỉ train head | 0,8460 | −0,1239 | −53 | kém hơn rõ | 0,8155 | 0,7786 | 0,9136 |
| T05 | B | + TrivialAugmentWide | **0,9736** | +0,0036 | +1,6 | tốt hơn (vượt 1 std) | 0,9474 | 0,9383 | 0,9882 |
| T06 | B | + CutMix (α = 1) | 0,9695 | −0,0004 | −0,2 | không phân biệt được | 0,9545 | 0,9264 | 0,9834 |
| T07 | C | label smoothing ε = 0,1 | 0,9698 | −0,0001 | 0,0 | không phân biệt được | 0,9476 | 0,9346 | 0,9854 |
| T08 | C | focal loss γ = 2 | 0,9700 | +0,0001 | 0,0 | không phân biệt được | 0,9361 | 0,9246 | 0,9877 |
| T12 | F | EMA trọng số (decay 0,998) | 0,9719 | +0,0020 | +0,8 | không phân biệt được | 0,9571 | 0,9301 | 0,9868 |
| T13 | kết hợp | T05 + T12 | 0,9726 | +0,0027 | +1,2 | vượt 1 std so với T00, nhưng không hơn T05 | 0,9524 | 0,9261 | 0,9882 |

**Nhận xét.**

- **Khởi tạo (A):** đóng băng backbone kém hẳn (−0,124, ≈ 53 std). Đặc trưng ImageNet cố định không đủ cho ảnh cỏ dại; tinh chỉnh toàn bộ là cần thiết. Đây là khác biệt lớn nhất trong Bước 2.
- **Augmentation (B):** TrivialAugment là yếu tố duy nhất vượt 1 std (+0,0036; +0,0046 so với trung bình 3 seed T00). CutMix không giúp. Giả thuyết: ảnh cỏ dại có vật thể nhỏ trên nền giống `Negatives`, cắt dán dễ bỏ mất cây mục tiêu mà nhãn vẫn trộn theo diện tích.
- **Loss (C):** label smoothing và focal không phân biệt được với CE về macro-F1. Focal làm F1 Chinee apple giảm (0,936 so với 0,952) và F1 Negatives tăng (0,988 so với 0,984): ngược với kỳ vọng "giúp lớp hiếm", nhưng chênh lệch theo lớp cũng nằm trong cỡ nhiễu 1 seed.
- **EMA (F):** +0,0020 < std: không phân biệt được. Không tốn thêm khi suy luận nên vẫn có thể giữ.
- **Kết hợp:** T13 (TrivialAugment + EMA) = 0,9726, tổng Δ riêng lẻ là +0,0056 nhưng Δ kết hợp chỉ +0,0027, và T13 − T05 = −0,0010 < std. **Hiệu ứng không cộng dồn**; T13 không phân biệt được với T05.
- **Kiểm tra lại bằng 3 seed (từ chung kết):** công thức T05 chạy 3 seed (`F01`, đánh giá 1 view lúc train) cho macro-F1 val **0,9744 ± 0,0016**, so với `T00` **0,9690 ± 0,0023**: Δ = +0,0054 lớn hơn cả hai std, nên trên val TrivialAugment là cải thiện thật. **Trên test thì không:** `F01rt` (cùng mô hình, 1 view) 0,9729 ± 0,0013 so với `T00` 0,9717 ± 0,0028, Δ = +0,0012 < std (mục 6). Lợi ích của công thức trên val nhiều khả năng một phần do chọn cấu hình trên chính tập val (Bước 2 chọn T05 vì val cao nhất).

---

## 5. Kết quả suy luận (Bước 3)

Mô hình `T05` seed 0 (macro-F1 val cao nhất Bước 2), không train lại. Độ trễ đo bằng `benchmark.py`: 10 lần warmup, `cuda.synchronize` trước và sau, 100 lần đo, batch 1, FP32 (trừ I08b/c), ảnh 224 trừ khi ghi khác, **không tính tiền xử lý**, RTX 4090, torch 2.11.0+cu128 (sheet `Inference`, `Latency`; biểu đồ `eval_out/inference_tradeoff.png`).

| exp_id | Phương pháp | K | macro-F1 val | Δ so với I00 | ECE val | p50 / p95 / p99 (ms) | Chi phí / I00 |
|---|---|---:|---:|---:|---:|---|---:|
| I00 | 1 view (mốc) | 1 | 0,9736 | – | 0,0101 | 3,97 / 4,10 / 4,15 | 1,00 |
| I01 | TTA lật ngang, gộp xác suất | 2 | 0,9750 | +0,0014 | 0,0081 | 3,97 / 4,00 / 4,03 | 1,00 |
| I02a | TTA 5 crop, gộp xác suất | 5 | 0,9743 | +0,0007 | 0,0088 | 3,95 / 4,03 / 4,05 | 1,00 |
| I02b | TTA 10 crop (5 + lật), gộp xác suất | 10 | 0,9750 | +0,0014 | 0,0064 | 4,85 / 4,93 / 4,93 | 1,22 |
| I02c | TTA 3 tỉ lệ 224/256/288, gộp xác suất | 3 | 0,9782 | +0,0046 | 0,0085 | 12,18 / 12,48 / 13,61 | 3,07 |
| I03a | như I01, gộp **logit** | 2 | 0,9750 | +0,0014 | 0,0081 | 3,97 / 4,00 / 4,03 | 1,00 |
| I03b | như I02b, gộp **logit** | 10 | 0,9759 | +0,0023 | 0,0083 | 4,85 / 4,93 / 4,93 | 1,22 |
| I03c | như I02c, gộp **logit** | 3 | 0,9765 | +0,0029 | 0,0088 | 12,18 / 12,48 / 13,61 | 3,07 |
| **I04_256** | **độ phân giải kiểm tra 256** | 1 | **0,9797** | **+0,0061** | 0,0066 | 3,93 / 3,96 / 3,98 | 0,99 |
| I04_288 | độ phân giải kiểm tra 288 | 1 | 0,9743 | +0,0007 | 0,0074 | 4,28 / 4,43 / 5,48 | 1,08 |
| I04_320 | độ phân giải kiểm tra 320 | 1 | 0,9720 | −0,0016 | 0,0071 | 3,95 / 3,99 / 4,02 | 0,99 |
| I05a | ensemble 3 seed `T00` | 3 | 0,9729 | (+0,0030 so với T00 seed 0) | 0,0095 | 11,72 / 12,07 / 12,23 | 2,95 |
| I05b | ensemble T05 + deit_small + swin_tiny | 3 | 0,9736 | 0,0000 | 0,0126 | 14,36 / 15,17 / 15,31 | 3,61 |
| I06 | trọng số EMA (`T12`) | 1 | 0,9719 | (+0,0020 so với T00 seed 0) | 0,0079 | = I00 | 1,00 |
| I07 | temperature scaling, T = 1,345 (khớp trên val) | 1 | 0,9736 | 0 | **0,0043** | = I00 | 1,00 |
| I08b | AMP (autocast FP16) | 1 | 0,9736 | 0 | 0,0104 | 5,46 / 5,74 / 7,09 | 1,37 |
| I08c | FP16 (`model.half()`) | 1 | 0,9736 | 0 | 0,0101 | 3,95 / 4,16 / 4,20 | 1,00 |

Tất cả là 1 mô hình, 1 seed; std nhiễu ~0,0023 dùng làm ngưỡng.

**Nhận xét.**

- **Độ phân giải kiểm tra 256** là phương pháp tốt nhất (+0,0061 ≈ 2,7 std) và **không tốn thêm** độ trễ ở batch 1. Phù hợp FixRes (slide trang 68): `RandomResizedCrop` lúc train làm vật thể trông to hơn ảnh val center-crop 224; test ở 256 (resize 293 → crop 256) bù lại một phần. Tăng tiếp lên 288/320 thì giảm, nên có một điểm tối ưu.
- **TTA:** lật ngang, 5 crop, 10 crop tăng 0,0007–0,0023, đều < 1 std, **không phân biệt được**. TTA 3 tỉ lệ +0,0029–0,0046 nhưng tốn 3,07 lần độ trễ, và phần lớn lợi ích có lẽ đến từ view 256 (I04_256 một mình đã tốt hơn).
- **Gộp xác suất hay logit:** giống hệt với 2 view, logit nhỉnh hơn với 10 crop (+0,0009) nhưng kém hơn với 3 tỉ lệ (−0,0017); mọi chênh lệch < std: **không có cách nào luôn tốt hơn**, khớp nhận định của slide.
- **Ensemble:** 3 seed `T00` tăng +0,0030 so với một seed (≈ 1,3 std) với chi phí 2,95 lần; 3 backbone khác nhau không tăng gì và ECE xấu hơn (0,0126), vì deit/swin yếu hơn convnext 0,015 kéo trung bình xuống.
- **EMA:** +0,0020 < std, miễn phí lúc suy luận.
- **Hiệu chuẩn:** temperature scaling (T = 1,345 > 1, mô hình hơi quá tự tin) giảm ECE val từ **0,0101 xuống 0,0043**; accuracy không đổi (đúng lý thuyết). Vì ECE đo trên chính tập vừa khớp T thì lạc quan, đo thêm chéo 2 nửa val (khớp trên nửa này, đo trên nửa kia): **0,0070**, vẫn thấp hơn 0,0101.
- **FP16 / AMP / gộp BN:** FP16 và AMP giữ nguyên macro-F1 (0,9736). Ở **batch 1, AMP chậm hơn FP32** (5,46 so với 3,97 ms, do chi phí ép kiểu); FP16 thuần ngang FP32. Ở **batch 32**, FP16 đạt 6.050 ảnh/s và AMP 4.632 ảnh/s so với FP32 2.170 ảnh/s (sheet `Latency`). Gộp BN **không áp dụng cho convnext_tiny** (dùng LayerNorm, không có cặp Conv–BN); hàm gộp BN đã kiểm tra đúng trên các mạng có BN (mục 2.1b).
- **Đánh đổi độ chính xác–độ trễ trên 4090:** ở batch 1 GPU còn dư sức, nên đưa K view vào cùng một batch gần như không tăng độ trễ (5 crop = 1 view; 10 crop chỉ 1,22 lần), khác với "chi phí ≈ K lần" của slide. Chi phí K lần chỉ hiện ra khi các lượt chạy nối tiếp (3 tỉ lệ khác kích thước, ensemble) hoặc khi xét thông lượng. Dữ liệu ủng hộ một phần kết luận của slide: thứ tốt nhất cho robot là thứ không tốn thêm (độ phân giải đã dò, EMA, FP16, temperature scaling); TTA nhiều tỉ lệ và ensemble đắt gấp ~3 lần mà lợi ích không vượt nhiễu, hợp chạy ngoại tuyến hơn.

---

## 6. Cấu hình tốt nhất và chung kết (Bước 4)

### 6.1 Cấu hình (chốt trên val, ghi ở `eval_out/final_config.json` trước khi mở test)

`F01` = `convnext_tiny.in12k_ft_in1k`, công thức `T00` + **`aug = "trivial"`** (TrivialAugmentWide sau RandomResizedCrop + lật ngang), 12 epoch, AdamW (1e-4 / 1e-3), wd 0,05, warmup 1 epoch + cosine, CE, batch 64, AMP; checkpoint theo macro-F1 val; suy luận **`I04_256`** (resize 293 → center crop 256, 1 view) + **temperature scaling khớp trên val của từng seed** (T = 1,278 / 1,295 / 1,204). Lý do: T05 có macro-F1 val cao nhất Bước 2; I04_256 hơn I00 +0,0061 > std 0,0023. Seed 0 của `F01` có cấu hình giống hệt `T05` seed 0 nên dùng lại (ghi trong `runs_logs/F01__seed0__summary.json`); seed 1, 2 train mới. Test chạy **một lần mỗi seed**; ô 4b từ chối chạy lại nếu file test đã có.

Kèm theo: `F01uncal` (như F01, chưa temperature scaling, cho I4a), `F01rt` (cùng mô hình, 1 view 224 + temperature scaling, cấu hình thời gian thực), mốc `T00` (công thức nền + 1 view, không TS).

### 6.2 Kết quả test (3 seed, `eval.py`; sheet `Final`)

| Cấu hình | macro-F1 val | **macro-F1 test** | top-1 test | balanced acc test | ECE test | recall Chinee | recall Snake |
|---|---|---|---|---|---|---|---|
| **F01** (T05 + I04_256 + TS) | 0,9783 ± 0,0016 | **0,9774 ± 0,0016** | **0,9820 ± 0,0003** | 0,9748 ± 0,0013 | **0,0050 ± 0,0026** | 0,9572 ± 0,0026 | 0,9592 ± 0,0221 |
| F01uncal (chưa TS) | 0,9783 ± 0,0016 | 0,9774 ± 0,0016 | 0,9820 ± 0,0003 | 0,9748 ± 0,0013 | 0,0070 ± 0,0009 | 0,9572 ± 0,0026 | 0,9592 ± 0,0221 |
| F01rt (T05 + 1 view + TS) | 0,9744 ± 0,0016 | 0,9729 ± 0,0013 | 0,9787 ± 0,0009 | 0,9722 ± 0,0022 | 0,0046 ± 0,0007 | 0,9351 ± 0,0111 | 0,9575 ± 0,0158 |
| **Mốc T00 + I00** | 0,9690 ± 0,0023 | 0,9717 ± 0,0028 | 0,9776 ± 0,0029 | 0,9748 ± 0,0034 | 0,0100 ± 0,0008 | 0,9410 ± 0,0102 | 0,9706 ± 0,0085 |

**So với mốc:** Δ macro-F1 test = 0,9774 − 0,9717 = **+0,0057**, lớn hơn std lớn hơn của hai nhóm (0,0028), nên là cải thiện vượt nhiễu, nhưng nhỏ (< 0,01). Tách ra: F01 − F01rt = +0,0045 (cùng mô hình, chỉ khác suy luận 256 so với 224) và F01rt − T00 = +0,0012 < std. **Phần lớn cải thiện trên test đến từ suy luận ở độ phân giải 256; công thức TrivialAugment không phân biệt được với nền trên test.** Recall Snake weed của F01 (0,959 ± 0,022) thấp hơn T00 (0,971 ± 0,009), chênh lệch nằm trong std của F01.

**Ổn định và hiệu chuẩn:** chênh macro-F1 val/test của F01 chỉ 0,0009; temperature scaling khớp trên val giảm ECE test 0,0070 → 0,0050 mà không đổi accuracy.

**Tự chấm phần I (`eval.py grade`, ngưỡng tạm thời):** I1 7/7 (top-1 98,20%), I2 4/5 (Δ = +0,0057 > s = 0,0028 nhưng < 0,01), I3 4/4, I4a 1/1, I4b 1/1, I5 2/2 (p95 4,1 ms) → **19/20** (`eval_out/grade_I.json`).

**Đối chiếu bài báo (trích dẫn, Olsen et al. 2019):** ResNet-50 95,7% (weighted average, 5 fold, ~100 epoch, augmentation mạnh); recall Chinee apple 88,5%, Snake weed 88,8%. Kết quả ở đây cao hơn (top-1 98,2%; recall 95,7% / 95,9%) nhưng **không so sánh trực tiếp được**: khác backbone và trọng số tiền huấn luyện (ImageNet-12k), khác định nghĩa accuracy, chỉ 1 fold, và split ngẫu nhiên không theo địa điểm (mục 8).

### 6.3 Theo từng lớp và ma trận nhầm lẫn

Precision / recall / F1 test (mean ± std 3 seed; sheet `PerClass`):

| Lớp | Số ảnh | F01 precision | F01 recall | F01 F1 | T00 F1 |
|---|---:|---|---|---|---|
| Chinee apple | 226 | 0,979 ± 0,003 | 0,957 ± 0,003 | 0,968 ± 0,001 | 0,956 ± 0,009 |
| Lantana | 213 | 0,987 ± 0,010 | 0,962 ± 0,012 | 0,975 ± 0,003 | 0,968 ± 0,004 |
| Parkinsonia | 207 | 0,987 ± 0,007 | 0,992 ± 0,003 | 0,990 ± 0,004 | 0,981 ± 0,008 |
| Parthenium | 205 | 1,000 ± 0,000 | 0,969 ± 0,006 | 0,984 ± 0,003 | 0,980 ± 0,004 |
| Prickly acacia | 213 | 0,949 ± 0,011 | 0,967 ± 0,005 | 0,958 ± 0,004 | 0,953 ± 0,004 |
| Rubber vine | 202 | 0,988 ± 0,003 | 0,983 ± 0,003 | 0,986 ± 0,001 | 0,982 ± 0,004 |
| Siam weed | 215 | 0,992 ± 0,005 | 0,992 ± 0,003 | 0,992 ± 0,004 | 0,982 ± 0,005 |
| Snake weed | 204 | 0,954 ± 0,024 | 0,959 ± 0,022 | 0,956 ± 0,008 | 0,958 ± 0,013 |
| Negatives | 1.822 | 0,985 ± 0,004 | 0,991 ± 0,001 | 0,988 ± 0,001 | 0,985 ± 0,003 |

Ma trận nhầm lẫn test của F01, cộng 3 seed: `eval_out/confusion_F01_test.png`. Cặp nhầm nhiều nhất (thật → dự đoán):

| Thật → dự đoán | Số ảnh (3 seed) | % ảnh của lớp thật |
|---|---:|---:|
| Negatives → Prickly acacia | 22 | 0,4% |
| Snake weed → Negatives | 20 | 3,3% |
| Prickly acacia → Negatives | 18 | 2,8% |
| Chinee apple → Negatives | 17 | 2,5% |
| Lantana → Negatives | 15 | 2,3% |
| Chinee apple → Snake weed | 12 | 1,8% |

- Lớp khó nhất là **Snake weed** (F1 0,956) và **Prickly acacia** (0,958), tiếp theo Chinee apple (0,968).
- **Kiểu lỗi chính là nhầm với `Negatives`**, không phải Chinee apple ↔ Snake weed như bài báo (bài báo: 3,4% Chinee → Snake, 4,1% chiều ngược lại; ở đây 1,8% Chinee → Snake). Prickly acacia bị nhầm cả hai chiều với `Negatives`.

**Phân tích ảnh bị đoán sai** (`eval_out/errors_chinee_snake.png`, seed 0 có 18 ảnh Chinee apple / Snake weed bị đoán sai): phần lớn bị đoán thành `Negatives`, nhiều ảnh với độ tin cậy cao (0,96–1,00). Đặc điểm thường gặp: (1) **ánh sáng cực đoan:** bóng đổ đậm che nửa khung, hoặc cháy sáng, ám tím/hồng; (2) **cây mục tiêu nhỏ hoặc lẫn** trong cỏ khô, lá rụng, thảm thực vật khác; (3) tán lá dày chồng lên nhau, không thấy rõ hình lá. Giả thuyết: (a) mô hình dựa nhiều vào màu và kết cấu tổng thể, nên ảnh có nền chiếm ưu thế bị kéo về lớp đông nhất (`Negatives`, 52% dữ liệu); (b) một số ảnh có thể **nhãn chưa chắc chắn** (ảnh Snake weed bị đoán `Negatives` với độ tin cậy 1,00 mà mắt thường khó thấy cây mục tiêu); (c) ColorJitter hoặc augmentation ánh sáng mạnh hơn có thể giúp (chưa thử, mục 2.5).

---

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất:** convnext_tiny (ImageNet-12k) + TrivialAugment + suy luận 256 + temperature scaling: **macro-F1 test 0,9774 ± 0,0016, top-1 0,9820 ± 0,0003**, hơn mốc `T00` + `I00` **+0,0057 macro-F1, vượt nhiễu** (std 0,0028) nhưng nhỏ.
- **Yếu tố đóng góp nhiều nhất là backbone:** trên val, chọn kiến trúc tạo khoảng 0,28 macro-F1 (mobilenetv3 0,689 → convnext_tiny 0,970); đổi công thức trong phạm vi đã thử chỉ ±0,004 (trừ đóng băng −0,12); đổi cách suy luận +0,006. Trên test, trong 0,0057 cải thiện so với mốc, khoảng 0,0045 đến từ suy luận 256 và 0,0012 (không phân biệt được với nhiễu) từ công thức.
- **Triển khai trên robot (ngân sách 30–100 ms/khung):** chọn **chính F01** (1 view ở 256, FP32 hoặc FP16): p95 **≈ 4,0 ms** ở batch 1 trên RTX 4090, macro-F1 test 0,9774. Không cần TTA hay ensemble: chúng đắt gấp ~3 lần mà lợi ích không vượt nhiễu. Nếu phần cứng robot yếu hơn nhiều (ví dụ Jetson; bài báo báo ResNet-50 trên TX2 cần 53,4 ms với TensorRT), phải đo lại trên thiết bị thật; lúc đó `F01rt` (224, ít hơn ~23% FLOPs so với 256) hoặc deit_small (nhanh nhất ở batch 1 trên 4090, kém 0,015 macro-F1) là phương án dự phòng.

## 8. Hạn chế và việc tiếp theo

- **Một fold** (fold 0), **3 seed** cho chung kết và mốc; ablation và so sánh backbone/suy luận chỉ **1 seed**. Các kết luận "tốt hơn" ở Bước 1–3 dựa trên std đo trên `T00` convnext_tiny, có thể không đúng cho backbone khác.
- **Split ngẫu nhiên, không theo địa điểm**: ảnh cùng một địa điểm hoặc cùng đợt chụp có thể nằm ở cả train và test, nên điểm test (98,2%) có thể **lạc quan** so với khi robot gặp địa điểm, mùa, ánh sáng hay góc chụp mới. Nên đánh giá thêm theo địa điểm.
- **Chọn trên val rồi báo cáo val:** công thức và cách suy luận đều được chọn trên cùng tập val, nên các số val của cấu hình được chọn bị lạc quan; điều này thấy rõ ở chỗ lợi ích của TrivialAugment vượt nhiễu trên val nhưng không trên test. Số test không bị ảnh hưởng vì chỉ chạy một lần sau khi đã chốt.
- **Cắt giảm do ngân sách** (mục 2.5): ablation 1 backbone, 6 thay vì 12 lần chạy, chưa thử train từ đầu, sampler cân bằng, CE có trọng số lớp, LR theo tầng.
- **Giả thuyết BatchNorm** cho nhóm ResNet/EfficientNet/MobileNet chưa kiểm chứng (cách kiểm chứng ở mục 3).
- **Độ trễ đo trên RTX 4090**, không phải phần cứng robot, không tính tiền xử lý (đọc, resize, chuẩn hoá ảnh trên CPU).
- **Rủi ro lệch phân phối:** mô hình nhạy với ánh sáng cực đoan và nền lấn át (mục 6.3); hiệu chuẩn (T khớp trên val) có thể không còn đúng khi miền ảnh đổi.
- **Việc tiếp theo:** chạy đủ 5 fold cho cấu hình cuối; kiểm chứng giả thuyết BN (tính lại thống kê BN, hoặc `RandomResizedCrop` với tỉ lệ tối thiểu lớn hơn); thử augmentation ánh sáng (ColorJitter) cho các lỗi do bóng đổ; dò độ phân giải train/test cùng nhau (FixRes đầy đủ); chưng cất convnext_tiny sang mạng nhẹ và đo trên Jetson.

## 9. Phụ lục

**Danh sách `exp_id`** (cấu hình đầy đủ trong `runs_logs/<exp_id>__seed<k>__config.json`, log theo epoch trong `__history.csv`, đường cong `curves/<exp_id>_<mota>.png`):

| exp_id | Nội dung | Seed |
|---|---|---|
| B01–B07 | resnet50, resnext50_32x4d, convnext_tiny, deit_small, swin_tiny, efficientnet_b0, mobilenetv3_large_100; công thức `T00` | 0 |
| T00 | convnext_tiny, công thức nền (seed 0 dùng lại B03) | 0, 1, 2 |
| T02, T05, T06, T07, T08, T12 | đóng băng; TrivialAugment; CutMix; label smoothing; focal; EMA | 0 |
| T13 | T05 + T12 | 0 |
| I00–I08 | phương pháp suy luận trên T05 seed 0 (I05a dùng T00 × 3, I05b dùng T05 + B04 + B05, I06 dùng T12) | – |
| F01 | T05 × 3 seed + I04_256 + TS (seed 0 dùng lại T05) | 0, 1, 2 |
| F01uncal, F01rt | F01 chưa TS; F01 với 1 view 224 + TS | 0, 1, 2 |

**Thứ tự chạy:** xem `README.md`. Bảng `Summary` (top 10 theo macro-F1 val) và các sheet khác nằm trong `results.xlsx`. Các dòng mean ± std của sheet `Final`, `PerClass`, `Summary` được tính lại từ `predictions/` với số chưa làm tròn sau khi chạy (bản notebook đầu tính từ số đã làm tròn, lệch ở chữ số thứ 4); ô Bước 4d trong notebook đã sửa để cho đúng kết quả này.
