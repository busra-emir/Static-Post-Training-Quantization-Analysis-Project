"""
metrics.py — Deney metrikleri

ÖLÇÜLEN METRİKLER:
──────────────────
1. Top-1 Accuracy (%)    : Modelin doğru tahmin yüzdesi
2. Accuracy Drop (ΔAcc)  : FP32 accuracy − INT8 accuracy (ne kadar kayıp var?)
3. ECE                   : Modelin güven skorlarının ne kadar kalibre olduğu
4. Model Size (MB)       : INT8 modelin disk boyutu
5. Inference Latency (ms): Bir batch'i işleme süresi

ECE (Expected Calibration Error) nedir?
─────────────────────────────────────────
Model bir kedi fotoğrafı için "%90 eminim" diyorsa,
gerçekten bu güven düzeyinde 100 tahmin yapsaydı 90 tanesi doğru olmalı.
ECE bu tutarlılığı ölçer. Düşük ECE = iyi kalibre model.
"""

import os
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ECE_BINS, LATENCY_WARMUP, DEVICE


def compute_accuracy(model: nn.Module, loader, device=DEVICE) -> float:
    """Top-1 accuracy (%) döndürür."""
    # Ensure fbgemm engine is active (required for quantized INT8 models)
    torch.backends.quantized.engine = "fbgemm"
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return round(100.0 * correct / total, 4)


def compute_accuracy_drop(fp32_acc: float, int8_acc: float) -> float:
    """
    ΔAcc = FP32 accuracy − INT8 accuracy
    Pozitif değer = INT8 daha kötü (beklenen)
    Sıfıra yakın = kalibrasyon başarılı
    """
    return round(fp32_acc - int8_acc, 4)


def compute_ece(model: nn.Module, loader, n_bins: int = ECE_BINS,
                device=torch.device("cpu"), **kwargs) -> float:
    """
    ECE = Σ_b (|B_b| / n) × |accuracy(B_b) − confidence(B_b)|

    Nasıl çalışır:
    - Tüm test görüntüleri üzerinde model çalıştırılır
    - Her görüntü için max softmax confidence ve tahmin doğruluğu alınır
    - Confidence değerleri 15 eşit aralıklı bin'e bölünür
    - Her bin'de ortalama confidence vs gerçek accuracy karşılaştırılır
    - Fark ne kadar küçükse model o kadar iyi kalibre
    """
    torch.backends.quantized.engine = "fbgemm"
    model.eval()
    all_confs, all_correct = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1)
            conf, pred = probs.max(dim=1)
            correct = (pred == labels.to(device)).float()
            all_confs.extend(conf.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_confs   = np.array(all_confs)
    all_correct = np.array(all_correct)
    total = len(all_confs)

    fraction_pos, mean_pred = calibration_curve(
        all_correct, all_confs, n_bins=n_bins, strategy="uniform"
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    weights = np.zeros(len(fraction_pos))
    for i in range(len(fraction_pos)):
        mask = (all_confs >= bin_edges[i]) & (all_confs < bin_edges[i + 1])
        weights[i] = mask.sum() / total

    ece = float(np.sum(weights * np.abs(fraction_pos - mean_pred)))
    return round(ece, 6)


def measure_model_size(model: nn.Module) -> float:
    """MB cinsinden model boyutu (geçici dosyaya yazarak ölçer)."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
    finally:
        os.remove(path)
    return round(size_mb, 3)


def measure_inference_latency(model: nn.Module, loader,
                               device=torch.device("cpu"),
                               warmup: int = LATENCY_WARMUP) -> float:
    """
    Ortalama inference latency (ms/batch).
    İlk `warmup` batch ısınma için atlanır.
    INT8 (fbgemm) modeller CPU'da çalışır.
    """
    model.eval()
    model.to(device)
    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            start = time.perf_counter()
            model(inputs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            if i >= warmup:
                times.append(elapsed)
            if i >= warmup + 20:
                break

    return round(float(np.mean(times)), 3) if times else float("nan")


def compute_all_metrics(fp32_model, int8_model, val_loader,
                        device=DEVICE, fp32_acc=None) -> dict:
    """
    Tek seferde tüm metrikleri hesaplar ve dict döndürür.
    INT8 model CPU'da çalışır (fbgemm backend).
    """
    cpu = torch.device("cpu")
    torch.backends.quantized.engine = "fbgemm"

    if fp32_acc is None:
        fp32_acc = compute_accuracy(fp32_model, val_loader, device)

    int8_acc  = compute_accuracy(int8_model, val_loader, cpu)
    ece       = compute_ece(int8_model, val_loader, device=cpu)
    size_mb   = measure_model_size(int8_model)
    latency   = measure_inference_latency(int8_model, val_loader, device=cpu)

    return {
        "fp32_accuracy":  round(fp32_acc, 4),
        "int8_accuracy":  round(int8_acc, 4),
        "accuracy_drop":  compute_accuracy_drop(fp32_acc, int8_acc),
        "ece":            ece,
        "model_size_mb":  size_mb,
        "latency_ms":     latency,
    }
