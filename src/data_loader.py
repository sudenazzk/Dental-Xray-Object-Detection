#!/usr/bin/env python3
"""
Diş Röntgeni Dataset Loader - Google Drive uyumlu
Yazar: Sudenaz Kabay
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import List, Tuple, Dict, Optional
import random

class DentalDatasetLoader:
    """Google Drive'dan diş röntgeni datasetini yükler"""
    
    def __init__(self, dataset_path: str, is_colab: bool = True):
        self.dataset_path = Path(dataset_path)
        self.is_colab = is_colab
        
        if is_colab and '/content/drive' not in str(dataset_path):
            print("⚠️  Google Colab'de çalışıyorsunuz ama dataset Google Drive'da değil")
        
        self.check_dataset_structure()
        self.classes = self.load_classes()
        print(f"✅ Dataset yüklendi: {len(self.classes)} sınıf")
    
    def check_dataset_structure(self) -> bool:
        required_dirs = [
            'train/images', 'train/labels',
            'valid/images', 'valid/labels', 
            'test/images', 'test/labels'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                print(f"❌ Eksik klasör: {dir_path}")
                return False
        
        print("✅ Dataset yapısı doğru")
        return True
    
    def load_classes(self) -> List[str]:
        yaml_path = self.dataset_path / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        
        return [
            'BDC-BDR',           # ID 0
            'Caries',            # ID 1 (Çürük)
            'Fractured Teeth',   # ID 2 (Kırık)
            'Healthy Teeth',     # ID 3 (Sağlıklı)
            'Impacted teeth',    # ID 4 (Gömülü)
            'Infection'          # ID 5 (Enfeksiyon)
        ]
    
    def get_image_paths(self, split: str = 'train') -> List[Path]:
        images_dir = self.dataset_path / split / 'images'
        if not images_dir.exists():
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(images_dir.glob(f'*{ext}'))
            image_paths.extend(images_dir.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    def get_label_path(self, image_path: Path, split: str = 'train') -> Optional[Path]:
        label_filename = image_path.stem + '.txt'
        label_path = self.dataset_path / split / 'labels' / label_filename
        
        if label_path.exists():
            return label_path
        return None
    
    def load_image(self, image_path: Path) -> np.ndarray:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Görsel yüklenemedi: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"❌ Görsel yükleme hatası: {e}")
            return np.zeros((640, 640, 3), dtype=np.uint8)
    
    def load_labels(self, label_path: Path) -> List[List[float]]:
        labels = []
        
        if not label_path.exists():
            return labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = list(map(float, line.split()))
                        if len(parts) >= 5:
                            labels.append(parts)
        except Exception as e:
            print(f"❌ Etiket yükleme hatası {label_path}: {e}")
        
        return labels

def test_data_loader():
    print("🧪 Data Loader Testi")
    print("=" * 50)
    
    test_path = "data"
    loader = DentalDatasetLoader(test_path, is_colab=False)
    
    train_images = loader.get_image_paths('train')
    if train_images:
        print(f"👁️  Örnek görsel: {train_images[0].name}")

if __name__ == "__main__":
    test_data_loader()
