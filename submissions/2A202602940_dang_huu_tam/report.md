# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Sinh viên:** Đặng Hữu Tâm · MSSV 2A202602940
**Repo:** https://github.com/tam253211-a11y/K4-Track4-Day2-DangHuuTam-2A202602940-Deeplearning-Advance
**Notebook:** [`code/lab_day2.ipynb`](code/lab_day2.ipynb) ([mở trên Colab](https://colab.research.google.com/github/tam253211-a11y/K4-Track4-Day2-DangHuuTam-2A202602940-Deeplearning-Advance/blob/main/submissions/2A202602940_dang_huu_tam/code/lab_day2.ipynb))

> Các ô ghi `[CHƯA ĐO]` sẽ được điền bằng số từ log chạy thật (`runs/`, `eval_out/`, `results.xlsx`). Không có số nào trong báo cáo lấy từ nguồn khác ngoài các số *trích dẫn* có ghi rõ nguồn.

---

## 1. Tóm tắt

`[CHƯA ĐO]`: bài toán, các thí nghiệm đã làm, cấu hình tốt nhất, macro-F1 và top-1 test (mean ± std, n seed), kết luận chính.

---

## 2. Dữ liệu và thiết lập

### 2.1 Dữ liệu

- DeepWeeds (Olsen et al., 2019): 17.509 ảnh RGB 256×256, 9 lớp (8 loài cỏ dại + `Negative`).
- Ảnh tải từ Zenodo (DOI 10.5281/zenodo.7939060), **MD5 `b7b30f96d466fba86016aa5a26606e0f` khớp**; giải nén được 17.509 file `.jpg`.
- Chia dữ liệu: **fold 0** của tác giả (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`), tải nguyên bản, không sửa (quy tắc S1–S6).

Kiểm tra chia dữ liệu (README mục 2.1), số từ `dataset.check_split`:

| Kiểm tra | Kết quả |
|---|---|
| Số ảnh train / val / test | 10.501 / 3.501 / 3.507 (tổng 17.509) |
| Tỉ lệ (%) | 60,0 / 20,0 / 20,0, khớp 60/20/20 |
| train∩val, train∩test, val∩test | 0 / 0 / 0 (rỗng) |
| Hợp ba tập | 17.509 ảnh |
| File trong CSV tồn tại trong `data/images/` | Thiếu 0 file |

Số ảnh mỗi lớp trong từng tập (fold 0) và đối chiếu Table 1 của bài báo:

| Lớp | Train | Val | Test | Tổng (đếm) | Bài báo | Chênh |
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

Nhận xét EDA:

- Tổng 17.509 ảnh khớp bài báo. Hai lớp lệch 1 ảnh so với Table 1 (Chinee Apple +1, Lantana −1), tổng vẫn bằng nhau: nhiều khả năng một ảnh được gán nhãn khác giữa bản CSV hiện hành và bảng trong bài báo. Chúng tôi dùng nguyên CSV của tác giả, không sửa (S1).
- **Mất cân bằng:** lớp nhiều nhất / ít nhất = 9.106 / 1.009 ≈ **9,0 lần**; `Negatives` chiếm **52,0%**. Một mô hình luôn đoán `Negatives` đã đạt ~52% top-1, nên macro-F1 là chỉ số chính. Tỉ lệ lớp giữa train, val, test gần như giống hệt nhau (chia có phân tầng).
- **Ảnh:** 500 ảnh train ngẫu nhiên đều là 256×256, RGB. Mean RGB của 200 ảnh ≈ (0,379; 0,393; 0,385), std ≈ (0,228; 0,228; 0,226): tối và ít đỏ hơn ImageNet (0,485; 0,456; 0,406). Vẫn dùng mean/std của trọng số timm vì tinh chỉnh từ trọng số ImageNet, các lớp đầu đã quen phân phối đó.
- Biểu đồ phân bố lớp: `eval_out/eda_class_distribution.png`; ảnh mẫu 4 ảnh/lớp: `eval_out/eda_samples.png`. `[CẦN BỔ SUNG]`: nhận xét bằng mắt (cặp lớp dễ nhầm, `Negatives` trông như thế nào).

### 2.1b Kiểm tra pipeline trước khi chạy thật (GUIDE mục 1.3)

| # | Kiểm tra | Kết quả |
|---|---|---|
| 1 | Cố định seed (random, numpy, torch, worker DataLoader) | Seed 0 chạy lại: cùng thứ tự file, cùng ảnh sau augmentation, cùng khởi tạo head; seed 1 thì khác |
| 2 | Loss CE ban đầu ≈ −ln(1/9) = 2,197 | 2,190 và 2,198 (hai phiên, batch ngẫu nhiên, `model.py`); 2,226 (batch val thật, resnet50) |
| 3 | Overfit batch nhỏ (18 ảnh, 2 ảnh/lớp, 100 bước) | Loss 2,179 → 0,0020; accuracy ở eval mode 1,00 |
| 4 | Ảnh sau augmentation (đã giải chuẩn hoá) khớp nhãn | Nhãn của 16 ảnh khớp CSV; ảnh xem tại `eval_out/check_augmentation.png` (kèm CutMix) |
| 5 | `model.train()` / `model.eval()` đúng lúc | Sau `evaluate()` mọi module ở eval; 2 lần forward ở eval giống nhau, ở train (dropout 0,5) khác nhau |

Kiểm tra cài đặt: focal loss γ=0 bằng CE, label smoothing ε=0 bằng CE (sai số < 1e-6); CutMix qua 50 lần thử: `lam` luôn bằng đúng diện tích ảnh gốc còn lại (kể cả khi hộp bị cắt ở biên), vùng dán lấy từ đúng ảnh `y_b`; Mixup và `mixed_loss = lam·CE(y_a) + (1−lam)·CE(y_b)` đúng; đóng băng backbone chỉ còn `fc.weight`, `fc.bias` được train và 0 lớp BN ở train mode; trọng số lớp chỉ tính từ train. Chạy thử `run(Config(...))` 1 epoch từ đầu đến cuối ra đủ `history.csv`, `best.pt`, file dự đoán val đúng định dạng `eval.py`, ảnh đường cong. Bộ test của repo: **38 test, OK**.

### 2.2 Chỉ số

Chỉ số chính là **macro-F1** (9 lớp, trọng số bằng nhau). Chỉ số phụ: top-1 accuracy, balanced accuracy, F1 theo lớp (bắt buộc Chinee apple và Snake weed), ECE 15 bin, độ trễ p50/p95/p99. mean ± std qua ≥ 3 seed, `ddof=1`. Mọi số test tính bằng `eval.py` (không sửa).

Val dùng cho mọi lựa chọn (backbone, siêu tham số, checkpoint, phương pháp suy luận, nhiệt độ T). Test chỉ chạy **một lần mỗi seed** ở Bước 4.

### 2.3 Công thức nền `T00`

| Thành phần | Giá trị |
|---|---|
| Khởi tạo | ImageNet (timm), head mới 9 lớp, tinh chỉnh toàn bộ |
| Đầu vào | Train: `RandomResizedCrop(224)` + lật ngang. Val/test: resize 256 → `CenterCrop(224)` |
| Chuẩn hoá | mean/std theo cấu hình trọng số timm |
| Optimizer | AdamW, LR backbone 1e-4, head 1e-3 |
| Weight decay | 0,05, không áp dụng cho norm và bias |
| Lịch LR | warmup 1 epoch + cosine |
| Loss | Cross-entropy |
| Batch size | 64 |
| Epoch | 12 (giống nhau cho mọi backbone) |
| Mixed precision | AMP, `channels_last` |
| Chọn checkpoint | epoch có macro-F1 val cao nhất (hòa lấy epoch sớm hơn) |

### 2.4 Phần cứng và phần mềm

- Google Colab, GPU **Tesla T4**.
- Python 3.13.15, PyTorch 2.11.0+cu130, timm 1.0.29.
- Kết quả (`runs/`, `predictions/`, `curves/`, `eval_out/`) lưu thẳng ra Google Drive (`MyDrive/K4-Day2-DeepWeeds`) để không mất khi phiên bị ngắt.
- Seed: sàng lọc (Bước 1–3) dùng seed 0; chung kết dùng seed 0, 1, 2. `cudnn.benchmark = True` nên kết quả lặp lại gần đúng, không trùng từng bit.

### 2.5 Ngân sách GPU và các cắt giảm

Thời gian 1 epoch (train + val, công thức nền, batch 64, AMP, 224) đo trên T4 bằng `train.measure_epoch_time` (có `cuda.synchronize`, đo đủ 1 epoch, không ngoại suy). Đo **ba lần ở ba phiên Colab khác nhau**; lần 3 là số trong `eval_out/epoch_timing.csv` và output ô ngân sách của notebook:

| Backbone (tag timm) | Params (M) | GMAC | Train lần 1 (s) | Train lần 2 (s) | Train lần 3 (s) | Val lần 3 (s) | Tổng/epoch lần 3 (s) | Bộ nhớ đỉnh (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50 (`a1_in1k`) | 23,53 | 4,09 | 40,9 | 42,0 | 41,2 | 8,4 | 49,5 | 3,8 |
| resnext50_32x4d (`a1h_in1k`) | 23,00 | 4,23 | 52,2 | 45,3 | 45,3 | 9,5 | 54,8 | 4,6 |
| convnext_tiny (`in12k_ft_in1k`) | 27,83 | 4,45 | 66,0 | 55,7 | 55,2 | 8,8 | 64,0 | 5,1 |
| deit_small_patch16_224 (`fb_in1k`) | 21,67 | 4,60 ² | 41,3 | 41,6 | 42,8 | 8,9 | 51,7 | 3,1 |
| swin_tiny_patch4_window7_224 (`ms_in1k`) | 27,53 | 4,49 | 65,5 | 71,2 | 71,2 | 10,4 | 81,6 | 5,6 |
| efficientnet_b0 (`ra_in1k`) | 4,02 | 0,38 | 68,7 | 41,0 | 41,1 | 7,4 | 48,4 | 3,5 |
| mobilenetv3_large_100 (`ra_in1k`) | 4,21 | 0,22 | 48,8 | 36,9 | 37,5 | 7,7 | 45,2 | 2,1 |

Params đếm khi đã thay head 9 lớp (vì vậy nhỏ hơn số của slide, vốn tính head 1000 lớp). ² GMAC của deit_small trong output ô kiểm tra `model.py` (Bước 0, chạy trên CPU) là 4,24 vì bộ đếm FLOPs của PyTorch không đếm kernel attention gộp trên CPU; trên GPU (lúc train) đếm được 4,60, khớp slide (4,6). Đã sửa `count_gmacs` để luôn đếm attention bằng matmul tường minh, mọi bảng dùng 4,60. Nhận xét:

- **Thời gian đo một lần không ổn định:** cùng backbone, các phiên chênh tới 40% (efficientnet_b0: 68,7 s ở lần 1 so với 41,0–41,1 s ở lần 2–3), có thể do GPU Colab dùng chung, nhiệt độ/xung nhịp, hoặc lần 1 còn chịu chi phí khởi động (tải trọng số, cuDNN chọn thuật toán). Vì vậy chỉ dùng các số này để **lập ngân sách**, không để kết luận backbone nào nhanh hơn; độ trễ được đo kỹ ở Bước 3 (warmup, ≥ 50 lần, p50/p95/p99).
- **GMAC không dự đoán được thời gian train:** efficientnet_b0 (0,38 GMAC) và mobilenetv3 (0,22 GMAC) ít hơn resnet50 (4,09 GMAC) 10–18 lần, nhưng thời gian train một epoch chỉ ngang hoặc nhanh hơn chút (41,1 s và 37,5 s so với 41,2 s ở lần 3). Lý do: depthwise conv tận dụng GPU kém, và với ảnh nhỏ thì phần đọc/giải mã/augmentation trên CPU chiếm đáng kể (slide trang 43: *FLOPs không phải độ trễ*).

Kế hoạch số lần chạy (12 epoch/lần): 7 backbone × 1 seed (Bước 1), 12 lần ablation trên 1 backbone (Bước 2), chung kết 3 seed và mốc `T00` thêm 2 seed (Bước 4). Ước tính theo thời gian đo lần 3:

| Hạng mục | Giờ GPU (T4) |
|---|---:|
| B: 7 backbone × 1 seed | 1,32 |
| T: 12 lần ablation (1 backbone) | 1,98 |
| F: chung kết + mốc | 0,83 |
| **Tổng** | **4,13** |
| Cộng 20% cho chạy hỏng | 4,96 |

Ước tính T và F tính theo thời gian của resnet50; backbone thực sự đi tiếp là convnext_tiny, chậm hơn ~1,3 lần (64,0 s so với 49,5 s mỗi epoch), nên T + F thực tế khoảng 3,6 giờ. Thực tế còn bị giới hạn bởi hạn mức GPU Colab miễn phí: phiên bị ngắt nhiều lần (B01, T00 seed 1, T02 phải chạy tiếp từ checkpoint). Tổng ~5 giờ GPU chia được thành vài phiên Colab; mọi lần chạy lưu checkpoint mỗi epoch ra Drive để tiếp tục khi phiên bị ngắt.

**Cắt giảm đã áp dụng** (theo thứ tự ưu tiên của GUIDE mục 7):

1. **Ablation Bước 2 chỉ trên 1 backbone** (chọn ở Bước 1 dựa trên macro-F1 val và độ trễ, xem mục 3). Hệ quả: kết luận về công thức huấn luyện chỉ chắc chắn cho backbone này, chưa kiểm chứng tính tổng quát sang backbone khác.
2. **Bước 2 rút gọn còn 6 ablation trên 4 trục** (A: đóng băng; B: CutMix, TrivialAugment; C: label smoothing, focal; F: EMA) thay vì 12 dự kiến, do hạn mức GPU miễn phí và hạn nộp. Bỏ: train từ đầu (A), lật dọc và ColorJitter (B), CE có trọng số lớp (C), sampler cân bằng (D), cùng LR cho backbone/head (E). Hệ quả: chưa trả lời được các câu hỏi của trục D, E và câu "train từ đầu có kịp không".
3. **Ablation chạy 1 seed**; 3 seed dành cho chung kết và mốc. Hệ quả: các Δ ở Bước 2 nhỏ hơn độ lệch giữa seed (đo ở Bước 4) được ghi là "không phân biệt được".
4. Số epoch: 12 (trong khoảng 10–15 của đề), giống nhau cho mọi cấu hình.

---

## 3. Kết quả so sánh backbone

Bảy backbone, cùng công thức nền `T00` (mục 2.3), cùng seed 0, cùng split, 12 epoch. Checkpoint chọn theo macro-F1 val. Độ trễ ở bước này là **đo sơ bộ** (batch 1, FP32, Tesla T4, 50 lần đo sau 10 lần warmup, có `cuda.synchronize`, không tính tiền xử lý); đo kỹ ở Bước 3.

| exp_id | Backbone (tag timm) | Params (M) | GMAC | macro-F1 val | top-1 val | F1 Chinee apple | F1 Snake weed | F1 val epoch 1 | Best epoch | Train/epoch (s) | p50 / p95 batch 1 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B03** | **convnext_tiny.in12k_ft_in1k** | 27,83 | 4,45 | **0,9702** | **0,9786** | 0,9569 | 0,9320 | 0,8766 | 11 | 57,0 | 6,76 / 8,14 |
| B05 | swin_tiny_patch4_window7_224.ms_in1k | 27,53 | 4,49 | 0,9612 | 0,9726 | 0,9108 | 0,9317 | 0,8448 | 12 | 69,5 | 11,49 / 16,03 |
| B04 | deit_small_patch16_224.fb_in1k | 21,67 | 4,60 | 0,9452 | 0,9612 | 0,8824 | 0,8986 | 0,8566 | 9 | 48,3 | 5,57 / 8,05 |
| B01 | resnet50.a1_in1k | 23,53 | 4,09 | 0,7904 | 0,8495 | 0,6535 | 0,7244 | — ¹ | 12 | 48,6 | 6,66 / 7,09 |
| B06 | efficientnet_b0.ra_in1k | 4,02 | 0,38 | 0,7515 | 0,8186 | 0,6305 | 0,7154 | 0,4048 | 12 | 43,2 | 8,03 / 8,51 |
| B02 | resnext50_32x4d.a1h_in1k | 23,00 | 4,23 | 0,7091 | 0,7649 | 0,6938 | 0,7852 | 0,1913 | 12 | 52,2 | 6,68 / 16,30 |
| B07 | mobilenetv3_large_100.ra_in1k | 4,21 | 0,22 | 0,6302 | 0,7375 | 0,6744 | 0,5417 | 0,3750 | 12 | 40,2 | 8,86 / 12,09 |

¹ B01 bị ngắt phiên Colab ở epoch 2 và chạy tiếp từ checkpoint epoch 2 (`last.pt`); epoch 1 nằm trong `runs/B01/seed0/history.csv`. Biểu đồ: `eval_out/backbones_tradeoff.png`; đường cong từng backbone: `curves/B0x_<ten>.png`.

**Mức nhiễu để đọc bảng.** Mỗi backbone mới chạy **1 seed**. Ở Bước 2, `T00` (convnext_tiny, cùng công thức) chạy 3 seed: macro-F1 val 0,9702 / 0,9661 / 0,9698, tức **0,9687 ± 0,0023** (std mẫu). Std này đo trên convnext_tiny; các backbone khác có thể dao động khác, nên chỉ coi là cỡ độ lớn của nhiễu.

**Nhận xét.**

1. **Hai nhóm tách biệt rất rõ.** Ba mạng dùng LayerNorm (convnext_tiny, swin_tiny, deit_small) đạt macro-F1 0,945–0,970; bốn mạng dùng BatchNorm (resnet50, resnext50, efficientnet_b0, mobilenetv3) chỉ 0,63–0,79. Khoảng cách 0,15–0,34 lớn hơn nhiễu (~0,002) hàng chục lần, nên đây là chênh lệch thật dù mới 1 seed.
2. **Trong nhóm đầu**, convnext_tiny hơn swin_tiny 0,009 và hơn deit_small 0,025. Cả hai lớn hơn std của convnext (0,0023) vài lần, nhưng vì swin/deit chỉ có 1 seed nên thứ hạng convnext > swin là **có khả năng**, chưa khẳng định chắc chắn.
3. **Hội tụ.** Nhóm LayerNorm đạt F1 0,84–0,88 ngay sau epoch 1 (đặc trưng tiền huấn luyện dùng được ngay); nhóm BatchNorm chỉ 0,19–0,40 sau epoch 1 và tới epoch 12 vẫn còn tăng (best epoch = 12 ở cả 4 mạng), tức chưa hội tụ trong 12 epoch.
4. **Quá khớp.** resnext50 có val loss thấp nhất ở epoch 7 rồi tăng nhẹ, trong khi train loss tiếp tục giảm (0,27 → 0,22) và val F1 đứng ở ~0,70: dấu hiệu quá khớp/lệch giữa train và val. efficientnet_b0 và mobilenetv3 có train loss ~0,21–0,22 nhưng val loss 0,56–0,77: khoảng cách train–val lớn. Nhóm LayerNorm có val loss vẫn giảm tới cuối (min ở epoch 11–12, trừ deit min ở epoch 12 nhưng best F1 ở epoch 9), không có quá khớp rõ.
5. **Giả thuyết cho nhóm BatchNorm (chưa kiểm chứng).** Đặc trưng tiền huấn luyện của các mạng này không kém: thử linear probe nhanh (đặc trưng đóng băng + hồi quy logistic, 360 ảnh train / 225 ảnh val, trên CPU) cho macro-F1 0,68 (resnet50.a1), 0,74 (efficientnet_b0), so với 0,76 (convnext_tiny). Vậy vấn đề nằm ở **quá trình tinh chỉnh**, không phải ở trọng số tải về. Giả thuyết: `RandomResizedCrop(224)` với tỉ lệ 8–100% làm thống kê BatchNorm (running mean/var) học trên ảnh bị phóng to khác xa ảnh val nguyên khung, nên ở eval mode mô hình lệch; LayerNorm không lưu thống kê nên không bị. Có thể kiểm chứng bằng cách tính lại thống kê BN trên ảnh train với transform của val (không train lại). Đây cũng là câu trả lời một phần cho câu hỏi "chênh lệch ResNet ↔ ConvNeXt đến từ kiến trúc hay công thức": với **cùng một công thức**, kiến trúc dùng BatchNorm chịu thiệt nặng; công thức nền này hợp với nhóm LayerNorm hơn.
6. **So với ImageNet.** Trên ImageNet các mạng cỡ ~25M tham số này chỉ cách nhau vài điểm top-1 (slide trang 41–42), còn trên DeepWeeds với công thức nền thì chênh lệch tới 0,26 macro-F1 giữa convnext_tiny và resnet50. Thứ hạng ImageNet **không** dự đoán được thứ hạng ở đây; công thức tinh chỉnh và loại chuẩn hoá quan trọng hơn. Lưu ý thêm: tag `convnext_tiny.in12k_ft_in1k` được tiền huấn luyện trên ImageNet-12k (lớn hơn ImageNet-1k của các mạng còn lại), một lợi thế cần ghi nhận khi so sánh.
7. **FLOPs không dự đoán được độ trễ.** efficientnet_b0 (0,38 GMAC) và mobilenetv3 (0,22 GMAC) có p50 8,0 và 8,9 ms, **chậm hơn** resnet50 (4,09 GMAC, 6,7 ms) và deit_small (4,60 GMAC, 5,6 ms) ở batch 1 trên T4: depthwise conv và nhiều lớp nhỏ tận dụng GPU kém, ở batch 1 thời gian bị chi phối bởi số lần gọi kernel chứ không phải số phép nhân. swin_tiny chậm nhất (11,5 ms) dù GMAC ngang convnext, do các phép chia/dịch cửa sổ. Thời gian train/epoch cũng không theo GMAC (mạng nhẹ chỉ nhanh hơn resnet50 ~10–17%). Một số p95 cao bất thường (resnext50 16,3 ms so với p50 6,7 ms) cho thấy đo một lần có nhiễu; Bước 3 đo lại kỹ hơn.
8. **Độ trễ không phải ràng buộc ở bước này:** mọi backbone đều có p95 batch 1 ≤ 16,3 ms trên T4, dưới xa ngân sách 30–100 ms/khung.

**Chọn backbone cho Bước 2 và 3: `convnext_tiny` (B03).** Lý do bằng số liệu: macro-F1 val cao nhất (0,9702, hơn swin_tiny 0,009 và hơn nhóm BatchNorm ≥ 0,18); F1 hai lớp khó cao nhất (Chinee apple 0,957, Snake weed 0,932); độ trễ batch 1 thuộc nhóm nhanh (p50 6,8 ms, ngang resnet50, nhanh hơn swin_tiny ~40%); train nhanh hơn swin_tiny ~18% mỗi epoch, quan trọng khi Bước 2 có ~15 lần chạy. Chỉ chọn 1 backbone do ngân sách GPU (mục 2.5).

## 4. Kết quả công thức huấn luyện

`[CHƯA ĐO]`: bảng ablation theo từng trục (khác `T00` ở điểm nào, Δ macro-F1 val, F1 lớp hiếm), so sánh Δ với std, kết hợp các yếu tố tốt (cộng dồn hay triệt tiêu). Ghi rõ có dùng cách tham lam theo trục hay không.

## 5. Kết quả suy luận

`[CHƯA ĐO]`: bảng phương pháp I00–I0x (macro-F1 val, top-1 val, ECE, p50/p95/p99 batch 1, thông lượng, chi phí tương đối), đường đánh đổi độ chính xác–độ trễ, ECE trước/sau temperature scaling (T khớp trên val). Điều kiện đo độ trễ: warmup 10 lần, `cuda.synchronize`, ≥ 50 lần đo, batch 1 và 32, GPU/dtype/độ phân giải ghi rõ.

## 6. Cấu hình tốt nhất

`[CHƯA ĐO]`: mô tả đầy đủ để tái lập, bảng chung kết (val và test, mean ± std, ≥ 3 seed) so với mốc `T00` + `I00`, F1/recall từng lớp (Chinee apple, Snake weed), ma trận nhầm lẫn test, phân tích ảnh bị đoán sai. Đối chiếu với số *trích dẫn* từ bài báo (ResNet-50 95,7%; Inception-v3 95,1%; Chinee apple 88,5%, Snake weed 88,8%), lưu ý điều kiện huấn luyện khác (≈100 epoch, augmentation mạnh).

## 7. Kết luận và khuyến nghị

`[CHƯA ĐO]`:
- Cấu hình nào tốt nhất? Tốt hơn mốc bao nhiêu, có vượt nhiễu không?
- Yếu tố nào đóng góp nhiều nhất: backbone, công thức huấn luyện hay suy luận?
- Triển khai trên robot với ngân sách 30–100 ms/khung: chọn cấu hình nào (p95 batch 1 và macro-F1 test)?

## 8. Hạn chế và việc tiếp theo

- Chỉ dùng **một fold** (fold 0); chung kết `[CHƯA ĐO]` seed.
- Fold của tác giả **chia ngẫu nhiên, không theo địa điểm**, nên điểm test có thể lạc quan so với khi robot gặp địa điểm/mùa mới.
- Ablation Bước 2 chỉ trên 1 backbone và 1 seed (mục 2.5).
- `[CHƯA ĐO]`: thí nghiệm thất bại, rủi ro lệch phân phối, việc làm tiếp nếu có thêm thời gian.

## 9. Phụ lục

`[CHƯA ĐO]`: danh sách `exp_id` và cấu hình đầy đủ (`runs/<exp_id>/config.json`), thứ tự chạy notebook.
