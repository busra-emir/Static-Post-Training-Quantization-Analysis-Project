import torch

# ── Dataset ───────────────────────────────────────────────────────────────────
# ImageNet-1K: PTQ literatüründe standart (BRECQ, PD-Quant, FIMA-Q)
# HuggingFace üzerinden erişilir, tam dataset indirilmez
NUM_CLASSES = 1000

# ImageNet normalization sabitleri (tüm torchvision pretrained modeller bunu kullanır)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Modeller ──────────────────────────────────────────────────────────────────
# Hoca önerisi: küçükten büyüğe model karşılaştırması
# Hepsi torchvision'da ImageNet pretrained ağırlıklarla gelir → eğitim gerekmez
#
# NOT: EfficientNet-B0 deneylerden çıkarıldı çünkü mimarisi (SiLU aktivasyonu +
# SqueezeExcitation çarpımı) PyTorch eager-mode static PTQ tarafından
# desteklenmiyor (aten::silu ve empty_strided QuantizedCPU'da yok).
# FX graph mode quantization gerekirdi, kapsam dışında.
MODELS = ["resnet18", "resnet50", "mobilenet_v2"]

# ── Deney parametreleri ───────────────────────────────────────────────────────
# Araştırma sorusu: kalibrasyon için kaç görüntü yeterli?
CALIB_SIZES = [10, 50, 100, 500, 1000, 5000]

# Her boyut için 5 farklı rastgele örnekleme → mean ± std hesaplanabilir
SEEDS = [42, 123, 456, 789, 1024]

# Toplam deney: 3 model × 6 boyut × 5 seed = 90 run

# ── Metrik parametreleri ──────────────────────────────────────────────────────
ECE_BINS       = 15    # Expected Calibration Error için bin sayısı
LATENCY_BATCH  = 64    # Latency ölçümünde batch boyutu
LATENCY_WARMUP = 3     # İlk 3 batch ısınma için atlanır

# ── Klasörler ─────────────────────────────────────────────────────────────────
RESULTS_RAW_DIR  = "results/raw"
RESULTS_AGG_DIR  = "results/aggregated"
FIGURES_DIR      = "figures"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
