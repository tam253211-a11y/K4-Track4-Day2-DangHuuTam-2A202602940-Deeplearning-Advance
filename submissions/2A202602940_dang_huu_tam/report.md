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
| 2 | Loss CE ban đầu ≈ −ln(1/9) = 2,197 | 2,190 (batch ngẫu nhiên, `model.py`) và 2,226 (batch val thật, resnet50) |
| 3 | Overfit batch nhỏ (18 ảnh, 2 ảnh/lớp, 100 bước) | Loss 2,179 → 0,0020; accuracy ở eval mode 1,00 |
| 4 | Ảnh sau augmentation (đã giải chuẩn hoá) khớp nhãn | Nhãn của 16 ảnh khớp CSV; ảnh xem tại `eval_out/check_augmentation.png` (kèm CutMix) |
| 5 | `model.train()` / `model.eval()` đúng lúc | Sau `evaluate()` mọi module ở eval; 2 lần forward ở eval giống nhau, ở train (dropout 0,5) khác nhau |

Kiểm tra cài đặt: focal loss γ=0 bằng CE, label smoothing ε=0 bằng CE (sai số < 1e-6); đóng băng backbone chỉ còn `fc.weight`, `fc.bias` được train và 0 lớp BN ở train mode; trọng số lớp chỉ tính từ train. Chạy thử `run(Config(...))` 1 epoch từ đầu đến cuối ra đủ `history.csv`, `best.pt`, file dự đoán val đúng định dạng `eval.py`, ảnh đường cong. Bộ test của repo: **38 test, OK**.

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

Thời gian 1 epoch (train + val, công thức nền, batch 64, AMP, 224) đo trên T4 bằng `train.measure_epoch_time` (có `cuda.synchronize`), lưu ở `eval_out/epoch_timing.csv`:

| Backbone (tag timm) | Params (M) | GMAC | Train/epoch (s) | Val (s) | Tổng/epoch (s) | Bộ nhớ đỉnh (GB) |
|---|---:|---:|---:|---:|---:|---:|
| resnet50 (`a1_in1k`) | 23,53 | 4,09 | 40,9 | 9,6 | 50,5 | 6,5 |
| resnext50_32x4d (`a1h_in1k`) | 23,00 | 4,23 | 52,2 | 9,6 | 61,9 | 4,6 |
| convnext_tiny (`in12k_ft_in1k`) | 27,83 | 4,45 | 66,0 | 15,0 | 81,0 | 5,1 |
| deit_small_patch16_224 (`fb_in1k`) | 21,67 | 4,24 | 41,3 | 7,2 | 48,5 | 3,1 |
| swin_tiny_patch4_window7_224 (`ms_in1k`) | 27,53 | 4,49 | 65,5 | 9,5 | 75,0 | 5,6 |
| efficientnet_b0 (`ra_in1k`) | 4,02 | 0,38 | 68,7 | 12,7 | 81,4 | 3,5 |
| mobilenetv3_large_100 (`ra_in1k`) | 4,21 | 0,22 | 48,8 | 10,2 | 59,0 | 2,1 |

Params đếm khi đã thay head 9 lớp (vì vậy nhỏ hơn số của slide, vốn tính head 1000 lớp). Thời gian đo đủ 1 epoch (không ngoại suy). Đáng chú ý: **GMAC không dự đoán được thời gian**: efficientnet_b0 (0,38 GMAC) và mobilenetv3 (0,22 GMAC) train một epoch không nhanh hơn resnet50 (4,09 GMAC) trên T4, vì depthwise conv tận dụng GPU kém và ở kích thước này phần đọc/giải mã ảnh cũng chiếm đáng kể (slide trang 43: *FLOPs không phải độ trễ*).

Kế hoạch số lần chạy (12 epoch/lần): 7 backbone × 1 seed (Bước 1), 12 lần ablation trên 1 backbone (Bước 2), chung kết 3 seed và mốc `T00` thêm 2 seed (Bước 4). Ước tính theo thời gian đo ở trên:

| Hạng mục | Giờ GPU (T4) |
|---|---:|
| B: 7 backbone × 1 seed | 1,52 |
| T: 12 lần ablation (1 backbone) | 2,02 |
| F: chung kết + mốc | 0,84 |
| **Tổng** | **4,38** |
| Cộng 20% cho chạy hỏng | 5,26 |

Ước tính T và F tính theo thời gian của resnet50; nếu backbone đi tiếp chậm hơn, số giờ tăng tương ứng (trường hợp chậm nhất ≈ 1,6 lần). Tổng ~5 giờ GPU chia được thành vài phiên Colab; mọi lần chạy lưu checkpoint mỗi epoch ra Drive để tiếp tục khi phiên bị ngắt.

**Cắt giảm đã áp dụng** (theo thứ tự ưu tiên của GUIDE mục 7):

1. **Ablation Bước 2 chỉ trên 1 backbone** (chọn ở Bước 1 dựa trên macro-F1 val và độ trễ, xem mục 3). Hệ quả: kết luận về công thức huấn luyện chỉ chắc chắn cho backbone này, chưa kiểm chứng tính tổng quát sang backbone khác.
2. **Ablation chạy 1 seed**; 3 seed dành cho chung kết và mốc. Hệ quả: các Δ ở Bước 2 nhỏ hơn độ lệch giữa seed (đo ở Bước 4) được ghi là "không phân biệt được".
3. Số epoch: 12 (trong khoảng 10–15 của đề), giống nhau cho mọi cấu hình.

---

## 3. Kết quả so sánh backbone

`[CHƯA ĐO]`: bảng trích từ sheet `Backbones`, biểu đồ macro-F1 theo độ trễ/tham số, nhận xét (hội tụ, quá khớp, thứ hạng so với ImageNet, FLOPs và độ trễ), lý do chọn backbone đi tiếp.

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
