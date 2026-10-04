# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

**Sinh viên:** Đặng Hữu Tâm · MSSV 2A202602940

| Sản phẩm | Đường dẫn |
|---|---|
| Báo cáo | [`report.md`](report.md) |
| Bảng so sánh (7 sheet) | [`results.xlsx`](results.xlsx) |
| Biểu đồ training (mỗi `exp_id` một ảnh) | [`curves/`](curves) |
| File dự đoán test/val (chung kết `F01`, mốc `T00`, mỗi seed) | [`predictions/`](predictions) |
| Code (khung `starter/` đã hoàn thiện) + notebook | [`code/`](code) |
| Bảng phụ, biểu đồ phân tích, kết quả `eval.py` | [`eval_out/`](eval_out) |
| Log từng lần chạy (`config.json`, `summary.json`, `history.csv`) | [`runs_logs/`](runs_logs) |

## Chạy lại

**Notebook:** [`code/lab_day2.ipynb`](code/lab_day2.ipynb). Notebook nộp kèm output của lần chạy thật.

- Colab: [mở notebook](https://colab.research.google.com/github/tam253211-a11y/K4-Track4-Day2-DangHuuTam-2A202602940-Deeplearning-Advance/blob/main/submissions/2A202602940_dang_huu_tam/code/lab_day2.ipynb) → Runtime → T4 GPU.
- Kaggle: New Notebook → File → Import Notebook → dán link GitHub của `code/lab_day2.ipynb`; Settings: GPU, bật Internet; chạy bằng *Save & Run All*.
- **Máy có GPU qua SSH (cách đã dùng cho kết quả nộp, RTX 4090 thuê trên vast.ai):**
  ```bash
  python -m pip install -q pandas numpy matplotlib pillow scikit-learn openpyxl timm nbconvert ipykernel
  wget -q -O lab_day2.ipynb https://raw.githubusercontent.com/tam253211-a11y/K4-Track4-Day2-DangHuuTam-2A202602940-Deeplearning-Advance/main/submissions/2A202602940_dang_huu_tam/code/lab_day2.ipynb
  jupyter nbconvert --to notebook --execute lab_day2.ipynb --output lab_day2_run.ipynb --ExecutePreprocessor.timeout=-1
  ```
  (chạy trong `tmux`; ~80 phút trên RTX 4090). Kết quả nằm trong thư mục repo được clone, `submission_artifacts.zip` gom sẵn các file nộp.

Ô cài đặt tự `git clone` repo này, tải ảnh từ Zenodo (kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`) và 4 file CSV fold 0 từ GitHub của tác giả. Kết quả ghi vào `runs/`, `predictions/`, `curves/`, `eval_out/`: trên Colab nối sang Google Drive, trên Kaggle nằm trong `/kaggle/working` (lưu bằng *Save Version*).

**Thứ tự chạy** (mỗi ô chạy lại được: lần train đã xong được bỏ qua, lần dở dang chạy tiếp từ checkpoint epoch cuối; file test đã có thì không chạy lại test):

| # | Ô notebook | Việc | GPU (T4, ước tính) |
|---|---|---|---|
| 0 | Cài đặt → Tải dữ liệu → `IMAGES_DIR` → Nơi lưu bền | Chuẩn bị, mỗi phiên mới đều chạy | ~5 phút |
| 1 | Bước 0 | Kiểm tra split (S1–S6), EDA, kiểm tra pipeline, test của repo, đo 1 epoch | ~15 phút |
| 2 | Bước 1a → 1b | 7 backbone `B01–B07`, bảng `Backbones` | ~1,5 giờ (T4) / ~15 phút (4090) |
| 3 | Bước 2a → 2b | Ablation `T00` (3 seed) + 6 ablation (`T02, T05–T08, T12`) + kết hợp `T13` trên `convnext_tiny`, bảng `Training` | ~2 giờ (T4) / ~20 phút (4090) |
| 4 | Bước 3a → 3b | Suy luận `I00–I08` trên val, độ trễ p50/p95/p99, bảng `Inference`, `Latency` | ~20 phút |
| 5 | Bước 4a → 4d | Chốt cấu hình trên val, `F01` × 3 seed, test một lần mỗi seed, `eval.py score/grade`, `Final`, `PerClass` | ~30–45 phút |
| 6 | Bước 5 | Sheet `Summary`, định dạng xlsx, kiểm tra `curves/`, gom sản phẩm | ~2 phút |

Có thể train một cấu hình từ dòng lệnh (từ thư mục `code/`): `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

Tính lại chỉ số từ file dự đoán (từ gốc repo):

```bash
python eval.py score --pred "submissions/2A202602940_dang_huu_tam/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01
python eval.py grade --final "submissions/2A202602940_dang_huu_tam/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602940_dang_huu_tam/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602940_dang_huu_tam/predictions/F01uncal_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

## Môi trường và seed

- **Kết quả nộp chạy trên 1 GPU NVIDIA GeForce RTX 4090** (vast.ai, 54 nhân CPU, `NUM_WORKERS = 12`). Python 3.12.14, PyTorch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.30; thêm numpy, pandas, scikit-learn, openpyxl, matplotlib. Các lần chạy thăm dò trên Colab T4 không dùng trong bảng kết quả (xem `report.md` mục 2.4).
- Seed: Bước 1–3 dùng seed 0; chung kết `F01` và mốc `T00` dùng seed 0, 1, 2. Seed cố định `random`, `numpy`, `torch` và worker DataLoader; không đổi cách chia (fold 0 của tác giả, không sửa CSV). `cudnn.benchmark = True` nên lặp lại gần đúng, không trùng từng bit.
- Không commit dữ liệu và checkpoint (`.gitignore` chặn `data/`, `*.zip`, `*.pt`, `runs/`). Log của mọi lần chạy (`config.json`, `summary.json`, `history.csv`, `steps.csv`) nằm trong `runs_logs/`.
- Trọng số tiền huấn luyện: timm (tag ghi trong sheet `Backbones`).

## Kết quả chính (test, 3 seed, `eval.py`)

| Cấu hình | macro-F1 | top-1 | ECE | recall Chinee apple / Snake weed |
|---|---|---|---|---|
| **F01**: convnext_tiny + TrivialAugment + test ở 256 + temperature scaling | **0,9774 ± 0,0016** | **0,9820 ± 0,0003** | 0,0050 ± 0,0026 | 0,957 / 0,959 |
| Mốc T00 + I00 | 0,9717 ± 0,0028 | 0,9776 ± 0,0029 | 0,0100 ± 0,0008 | 0,941 / 0,971 |

`eval.py grade` (đề xuất): 19/20 điểm phần I. Độ trễ p95 batch 1 của F01 ≈ 4,0 ms (RTX 4090, FP32).
