#!/usr/bin/env python3
"""
Diş Röntgeni Projesi - Yardımcı Fonksiyonlar
Google Colab uyumlu
"""

import os
import json
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

def setup_colab_environment():
    print("🔧 Google Colab ortamı hazırlanıyor...")
    
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            print("✅ Google Drive bağlandı")
        else:
            print("✅ Google Drive zaten bağlı")
    except:
        print("⚠️  Google Colab'de değilsiniz veya Drive bağlantısı başarısız")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except:
        print("📦 Ultralytics yükleniyor...")
        os.system('pip install -q ultralytics')
    
    print("✅ Ortam hazırlandı")

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✅ Random seed ayarlandı: {seed}")
    except:
        pass

def load_config(config_path: str = 'config.yaml') -> Dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Konfigürasyon yüklendi: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Konfigürasyon yüklenemedi: {e}")
        return {}

def save_results(results: Dict, filename: str = 'training_results.json'):
    results['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Sonuçlar kaydedildi: {filename}")
    return filename

def copy_model_to_drive(model_path: str, drive_path: str = None):
    if drive_path is None:
        drive_path = '/content/drive/MyDrive/yolov8_dental_opg_best.pt'
    
    try:
        if os.path.exists(model_path):
            shutil.copy(model_path, drive_path)
            print(f"✅ Model Drive'a kopyalandı: {drive_path}")
            return True
        else:
            print(f"❌ Model dosyası bulunamadı: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Kopyalama hatası: {e}")
        return False

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    if iteration == total: 
        print()

def test_utils():
    print("🧪 Utils Testi")
    print("=" * 50)
    
    set_random_seed(42)
    config = load_config()
    print(f"Config yüklendi: {bool(config)}")
    
    print("✅ Utils testi tamamlandı")

if __name__ == "__main__":
    test_utils()
