"""
quantization.py — FP32 → INT8 PTQ Pipeline

AŞAMALAR (sırayla):
───────────────────
1. PREPARE   : Modele "observer" ekle — aktivasyon aralıklarını kaydetmeye hazırla
2. CALIBRATE : Kalibrasyon görüntülerini modelden geçir — her katmanın min/max değerlerini ölç
3. CONVERT   : Ölçülen değerlere göre scale/zero-point hesapla, FP32 → INT8 dönüştür

NEDEN DEEPCOPY:
───────────────
Her deney (her seed, her n) için aynı FP32 modelden başlamamız lazım.
deepcopy olmadan model bir kez dönüştürülünce bir daha FP32'ye dönemez.
"""

import copy
import torch
import torch.nn as nn
import torch.quantization as tq


def _get_fusion_layers(model_name: str) -> list:
    """
    BN + Conv katmanlarını birleştirme listesi.

    Neden birleştiriyoruz?
    Conv → BN ayrı ayrı quantize edilirse hata birikir.
    Birleştirilince tek katman olarak daha temiz quantize edilir.

    ResNet-18: her blokta conv1+bn1+relu ve conv2+bn2 var (BasicBlock)
    ResNet-50: her blokta conv1+bn1+relu, conv2+bn2+relu, conv3+bn3 var (Bottleneck)
    EfficientNet / MobileNetV2: karmaşık iç yapı, fusion yapılmaz
    """
    if model_name == "resnet18":
        return [
            ["conv1", "bn1", "relu"],
            ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"],
            ["layer1.0.conv2", "layer1.0.bn2"],
            ["layer1.1.conv1", "layer1.1.bn1", "layer1.1.relu"],
            ["layer1.1.conv2", "layer1.1.bn2"],
            ["layer2.0.conv1", "layer2.0.bn1", "layer2.0.relu"],
            ["layer2.0.conv2", "layer2.0.bn2"],
            ["layer2.0.downsample.0", "layer2.0.downsample.1"],
            ["layer2.1.conv1", "layer2.1.bn1", "layer2.1.relu"],
            ["layer2.1.conv2", "layer2.1.bn2"],
            ["layer3.0.conv1", "layer3.0.bn1", "layer3.0.relu"],
            ["layer3.0.conv2", "layer3.0.bn2"],
            ["layer3.0.downsample.0", "layer3.0.downsample.1"],
            ["layer3.1.conv1", "layer3.1.bn1", "layer3.1.relu"],
            ["layer3.1.conv2", "layer3.1.bn2"],
            ["layer4.0.conv1", "layer4.0.bn1", "layer4.0.relu"],
            ["layer4.0.conv2", "layer4.0.bn2"],
            ["layer4.0.downsample.0", "layer4.0.downsample.1"],
            ["layer4.1.conv1", "layer4.1.bn1", "layer4.1.relu"],
            ["layer4.1.conv2", "layer4.1.bn2"],
        ]

    elif model_name == "resnet50":
        return [
            ["conv1", "bn1", "relu"],
            ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"],
            ["layer1.0.conv2", "layer1.0.bn2"],
            ["layer1.0.conv3", "layer1.0.bn3"],
            ["layer1.0.downsample.0", "layer1.0.downsample.1"],
            ["layer1.1.conv1", "layer1.1.bn1", "layer1.1.relu"],
            ["layer1.1.conv2", "layer1.1.bn2"],
            ["layer1.1.conv3", "layer1.1.bn3"],
            ["layer1.2.conv1", "layer1.2.bn1", "layer1.2.relu"],
            ["layer1.2.conv2", "layer1.2.bn2"],
            ["layer1.2.conv3", "layer1.2.bn3"],
            ["layer2.0.conv1", "layer2.0.bn1", "layer2.0.relu"],
            ["layer2.0.conv2", "layer2.0.bn2"],
            ["layer2.0.conv3", "layer2.0.bn3"],
            ["layer2.0.downsample.0", "layer2.0.downsample.1"],
            ["layer2.1.conv1", "layer2.1.bn1", "layer2.1.relu"],
            ["layer2.1.conv2", "layer2.1.bn2"],
            ["layer2.1.conv3", "layer2.1.bn3"],
            ["layer2.2.conv1", "layer2.2.bn1", "layer2.2.relu"],
            ["layer2.2.conv2", "layer2.2.bn2"],
            ["layer2.2.conv3", "layer2.2.bn3"],
            ["layer2.3.conv1", "layer2.3.bn1", "layer2.3.relu"],
            ["layer2.3.conv2", "layer2.3.bn2"],
            ["layer2.3.conv3", "layer2.3.bn3"],
            ["layer3.0.conv1", "layer3.0.bn1", "layer3.0.relu"],
            ["layer3.0.conv2", "layer3.0.bn2"],
            ["layer3.0.conv3", "layer3.0.bn3"],
            ["layer3.0.downsample.0", "layer3.0.downsample.1"],
            ["layer3.1.conv1", "layer3.1.bn1", "layer3.1.relu"],
            ["layer3.1.conv2", "layer3.1.bn2"],
            ["layer3.1.conv3", "layer3.1.bn3"],
            ["layer3.2.conv1", "layer3.2.bn1", "layer3.2.relu"],
            ["layer3.2.conv2", "layer3.2.bn2"],
            ["layer3.2.conv3", "layer3.2.bn3"],
            ["layer3.3.conv1", "layer3.3.bn1", "layer3.3.relu"],
            ["layer3.3.conv2", "layer3.3.bn2"],
            ["layer3.3.conv3", "layer3.3.bn3"],
            ["layer3.4.conv1", "layer3.4.bn1", "layer3.4.relu"],
            ["layer3.4.conv2", "layer3.4.bn2"],
            ["layer3.4.conv3", "layer3.4.bn3"],
            ["layer3.5.conv1", "layer3.5.bn1", "layer3.5.relu"],
            ["layer3.5.conv2", "layer3.5.bn2"],
            ["layer3.5.conv3", "layer3.5.bn3"],
            ["layer4.0.conv1", "layer4.0.bn1", "layer4.0.relu"],
            ["layer4.0.conv2", "layer4.0.bn2"],
            ["layer4.0.conv3", "layer4.0.bn3"],
            ["layer4.0.downsample.0", "layer4.0.downsample.1"],
            ["layer4.1.conv1", "layer4.1.bn1", "layer4.1.relu"],
            ["layer4.1.conv2", "layer4.1.bn2"],
            ["layer4.1.conv3", "layer4.1.bn3"],
            ["layer4.2.conv1", "layer4.2.bn1", "layer4.2.relu"],
            ["layer4.2.conv2", "layer4.2.bn2"],
            ["layer4.2.conv3", "layer4.2.bn3"],
        ]

    # EfficientNet ve MobileNetV2 için fusion yapılmaz
    return []


def full_ptq_pipeline(
    fp32_model: nn.Module,
    model_name: str,
    calib_loader,
    device: torch.device,
) -> nn.Module:
    """
    Tam PTQ Pipeline: 3 adımda FP32 → INT8

    Adım 1 — PREPARE:
      - Modelin kopyasını al (orijinal bozulmasın)
      - BN+Conv katmanlarını birleştir (fusion)
      - Her katmana "observer" ekle (aktivasyon takip edecek)

    Adım 2 — CALIBRATE:
      - Kalibrasyon görüntülerini modelden geçir
      - Observer'lar her katmanın min/max değerlerini kaydeder
      - Gradient hesabı kapalı (öğrenme yok, sadece ölçüm)

    Adım 3 — CONVERT:
      - Observer'ların topladığı istatistiklerden scale/zero-point hesapla
      - FP32 ağırlık ve aktivasyonları INT8'e dönüştür
      - Sonuç: 4x daha küçük, 2-4x daha hızlı model
    """
    # ── Adım 1: Prepare ───────────────────────────────────────────────────────
    # Set fbgemm backend globally before any quantization ops
    torch.backends.quantized.engine = "fbgemm"

    m = copy.deepcopy(fp32_model)   # orijinal FP32 modeli koru
    m.eval()

    # BN + Conv fusion (hata azaltımı)
    fusion_layers = _get_fusion_layers(model_name)
    if fusion_layers:
        try:
            tq.fuse_modules(m, fusion_layers, inplace=True)
        except Exception as e:
            print(f"  [UYARI] Fusion kısmen başarısız: {e}")

    # fbgemm: x86 CPU için optimize INT8 backend
    m.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(m, inplace=True)

    # ── Adım 2: Calibrate ─────────────────────────────────────────────────────
    # Observer'lar aktivasyon aralıklarını ölçer
    # Bu adımda model HİÇBİR ŞEY ÖĞRENMEZ, sadece ölçüm yapılır
    m.eval()
    m.to(device)
    with torch.no_grad():
        for inputs, _ in calib_loader:
            m(inputs.to(device))

    # ── Adım 3: Convert ───────────────────────────────────────────────────────
    # fbgemm CPU'da çalışır
    m.cpu()
    tq.convert(m, inplace=True)

    return m
