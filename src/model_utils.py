"""
model_utils.py — Model yükleme ve değerlendirme

NASIL ÇALIŞIR:
─────────────
Bu dosyada EĞİTİM YOKTUR.

torchvision'daki modeller zaten ImageNet üzerinde eğitilmiş gelir:
  resnet18      → ImageNet Top-1: ~69.8%
  resnet50      → ImageNet Top-1: ~76.1%
  efficientnet_b0 → ImageNet Top-1: ~77.7%
  mobilenet_v2  → ImageNet Top-1: ~71.9%

Biz sadece bu hazır modelleri yükleyip PTQ uyguluyoruz.
Eğitim gerekmediği için Notebook 01 çok hızlı çalışır (~2 dk).
"""

import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.models.quantization as tv_qmodels
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NUM_CLASSES, RESULTS_AGG_DIR, DEVICE


class _SiLUWrapper(nn.Module):
    """
    SiLU katmanını DeQuant → SiLU → Quant ile sarar.

    NEDEN GEREKLİ:
    ──────────────
    PyTorch'un QuantizedCPU backend'inde 'aten::silu' kernel'i YOK.
    EfficientNet baştan sona SiLU kullanıyor → INT8 inference çakılır.
    Bu wrapper SiLU'yu float'a çevirip çalıştırıp tekrar quantize ediyor.
    Sonuç: SiLU float'ta çalışır, geri kalan tüm pipeline INT8 kalır.
    """
    def __init__(self):
        super().__init__()
        self.dequant = torch.quantization.DeQuantStub()
        self.silu = nn.SiLU()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.dequant(x)
        x = self.silu(x)
        x = self.quant(x)
        return x


def _wrap_silu_with_quant_stubs(model: nn.Module) -> nn.Module:
    """
    Modeldeki tüm nn.SiLU katmanlarını _SiLUWrapper ile değiştirir.
    EfficientNet'te ~50 SiLU vardır, hepsi tek seferde değişir.
    """
    def replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.SiLU):
                setattr(module, name, _SiLUWrapper())
            else:
                replace(child)
    replace(model)
    return model


def load_model(model_name: str, device=DEVICE) -> nn.Module:
    """
    ImageNet pretrained ağırlıklarla modeli yükler.
    Eğitim gerekmez — torchvision'dan hazır gelir.

    NEDEN quantization versiyonları?
    ─────────────────────────────────
    PyTorch static PTQ için modelin başında QuantStub,
    sonunda DeQuantStub olması şarttır. Standart tv_models.*
    bunlara sahip değildir → INT8 layer float tensor alır → hata.

    torchvision.models.quantization.* bu stublara sahiptir.
    quantize=False: yapıyı hazır yükle, PTQ'yu biz uygularız.

    Desteklenen modeller:
      resnet18       → 11M parametre, hafif
      resnet50       → 25M parametre, güçlü
      efficientnet_b0 → 5M parametre, verimli  (QuantWrapper ile)
      mobilenet_v2   → 3.4M parametre, mobil
    """
    print(f"  {model_name} yükleniyor (ImageNet pretrained)...")

    if model_name == "resnet18":
        # quantize=False: PTQ hazır yapı, ama henüz quantize etme
        model = tv_qmodels.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1,
            quantize=False,
        )

    elif model_name == "resnet50":
        model = tv_qmodels.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1,
            quantize=False,
        )

    elif model_name == "mobilenet_v2":
        model = tv_qmodels.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1,
            quantize=False,
        )

    elif model_name == "efficientnet_b0":
        # EfficientNet için iki düzeltme gerekli:
        # 1) SiLU katmanları QuantizedCPU'da yok → DeQuant→SiLU→Quant ile sar
        # 2) Modelin giriş/çıkışına QuantStub/DeQuantStub ekle (QuantWrapper)
        base = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        base = _wrap_silu_with_quant_stubs(base)
        model = torch.quantization.QuantWrapper(base)

    else:
        raise ValueError(
            f"Bilinmeyen model: '{model_name}'\n"
            f"Geçerli seçenekler: resnet18, resnet50, efficientnet_b0, mobilenet_v2"
        )

    model = model.to(device)
    model.eval()
    print(f"  ✓ {model_name} yüklendi")
    return model


def evaluate_model(model: nn.Module, loader, device=DEVICE) -> float:
    """
    Modelin Top-1 accuracy'sini ölçer.

    KULLANIM AMACI:
      - FP32 modelin baseline accuracy'si
      - INT8 modelin kalibrasyon sonrası accuracy'si
      - İkisi karşılaştırılarak "accuracy drop" hesaplanır
    """
    model.eval()
    correct = total = 0

    # Değerlendirme sırasında gradient hesabına gerek yok → hızlandırır
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def save_fp32_baselines(results: dict) -> pd.DataFrame:
    """
    FP32 baseline accuracy'leri CSV olarak kaydeder.
    Örnek: {'resnet18': 69.8, 'resnet50': 76.1, ...}
    """
    os.makedirs(RESULTS_AGG_DIR, exist_ok=True)
    path = os.path.join(RESULTS_AGG_DIR, "fp32_baselines.csv")
    df = pd.DataFrame(list(results.items()), columns=["model", "fp32_accuracy"])
    df.to_csv(path, index=False)
    print(f"  ✓ FP32 baseline'lar kaydedildi: {path}")
    return df
