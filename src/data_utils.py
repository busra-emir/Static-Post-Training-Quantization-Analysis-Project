"""
data_utils.py — ImageNet-1K veri yükleme ve kalibrasyon seti oluşturma

NASIL ÇALIŞIR:
─────────────
1. ImageNet validation seti HuggingFace üzerinden stream edilir
   → Tüm dataset (150GB) indirilmez, sadece ihtiyaç kadar çekilir

2. get_val_loader():
   → Modelin FP32 accuracy'sini ölçmek için kullanılır
   → Tüm 50.000 validation görüntüsü üzerinde çalışır

3. get_calibration_subset(n, seed):
   → INT8 dönüşümü için scale/zero-point hesaplamasında kullanılır
   → Validation setinden n adet görüntü rastgele seçilir
   → ARAŞTIRMA SORUSU: n=10 yeterli mi? n=5000 şart mı?

NOT: HuggingFace'de "imagenet-1k" dataset'ine erişmek için:
   1. huggingface.co'da hesap aç
   2. imagenet-1k sayfasında "Access repository" tıkla
   3. Colab'da: huggingface-cli login komutu ile giriş yap
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import IMAGENET_MEAN, IMAGENET_STD, LATENCY_BATCH


def _set_seeds(seed: int):
    """Tekrar üretilebilirlik için tüm random seed'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ImageNet için standart transform
# Resize(256) → CenterCrop(224): torchvision pretrained modellerin beklediği format
_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HFImageNetDataset(Dataset):
    """
    HuggingFace'den çekilen ImageNet verilerini PyTorch Dataset'e çevirir.
    Her örnek: (224×224 tensor, int label) döndürür.
    """
    def __init__(self, records: list, transform=_VAL_TRANSFORM):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = item["image"]
        # Bazı ImageNet görüntüleri grayscale olabilir → RGB'ye çevir
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = item["label"]
        return self.transform(image), label


def _load_hf_dataset(n_samples: int, seed: int = 0):
    """
    HuggingFace'den ImageNet validation setinden n_samples adet örnek çeker.
    Streaming mod: tüm dataset indirilmez.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets kütüphanesi bulunamadı.\n"
            "Colab'da çalıştır: !pip install datasets -q"
        )

    # Streaming mod: sadece ihtiyaç kadar görüntü çekilir
    # "ILSVRC/imagenet-1k" — gated dataset, HuggingFace'de erişim onayı gerekir
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
    )

    # n_samples kadar örnek topla
    _set_seeds(seed)
    records = []

    # Önce 10×n_samples adet örnek al, sonra rastgele n_samples seç
    # (streaming'de shuffle sınırlı, bu yaklaşım daha dengeli dağılım sağlar)
    buffer_size = min(n_samples * 10, 50000)
    buffer = []
    for i, item in enumerate(ds):
        buffer.append(item)
        if len(buffer) >= buffer_size:
            break

    # Buffer'dan rastgele seç
    indices = random.sample(range(len(buffer)), min(n_samples, len(buffer)))
    records = [buffer[i] for i in indices]

    return records


def get_val_loader(batch_size: int = 64, max_samples: int = 10000) -> DataLoader:
    """
    ImageNet validation setini yükler.
    max_samples: bellek tasarrufu için kaç örnek kullanılacağı
                 (None = tüm 50k, 10000 = hızlı değerlendirme için)

    KULLANIM AMACI: FP32 ve INT8 modelin Top-1 accuracy'sini ölçmek
    """
    print(f"  Validation seti yükleniyor ({max_samples} örnek)...")
    records = _load_hf_dataset(n_samples=max_samples, seed=0)
    dataset = HFImageNetDataset(records)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def get_calibration_subset(n: int, seed: int) -> DataLoader:
    """
    Kalibrasyon seti: validation setinden n adet görüntü seçer.

    KULLANIM AMACI:
    INT8 dönüşümü sırasında her katmanın aktivasyon aralığını ölçmek için
    bu görüntüler modelden geçirilir. Model bu görüntülerden ÖĞRENMEZ,
    sadece "bu katman normalde ne kadar değer üretiyor?" sorusunu cevaplar.

    n = 10   → çok az ölçüm → kaba scale değerleri → accuracy düşer
    n = 5000 → çok ölçüm   → hassas scale değerleri → accuracy korunur
    """
    print(f"  Kalibrasyon seti oluşturuluyor (n={n}, seed={seed})...")
    _set_seeds(seed)
    records = _load_hf_dataset(n_samples=n, seed=seed)
    dataset = HFImageNetDataset(records)
    return DataLoader(dataset, batch_size=min(n, 64), shuffle=False,
                      num_workers=2, pin_memory=True)
