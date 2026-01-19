#!/usr/bin/env python3
"""
Diş Röntgeni Nesne Tespiti - YOLOv8 Model Eğitim Script'i
Yazar: Sudenaz Kabay
Google Colab ile uyumlu
"""

import argparse
import os
import sys
from ultralytics import YOLO
from pathlib import Path

def setup_colab_environment():
    """Google Colab ortamını ayarla"""
    print("🔧 Colab ortamı kontrol ediliyor...")
    
    # Drive bağlantısını kontrol et
    if os.path.exists('/content/drive'):
        print("✅ Google Drive bağlı")
    else:
        print("⚠️  Google Drive bağlı değil")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive bağlandı")
        except:
            print("❌ Google Drive bağlanamadı")
    
    # Gerekli kütüphaneleri kontrol et
    try:
        import ultralytics
        print(f"✅ Ultralytics yüklü: {ultralytics.__version__}")
    except ImportError:
        print("📦 Ultralytics yükleniyor...")
        os.system('pip install -q ultralytics')
        print("✅ Ultralytics yüklendi")

def prepare_dataset_for_training(dataset_path):
    """Dataset'i YOLO formatına hazırla"""
    print(f"\n📁 Dataset hazırlanıyor: {dataset_path}")
    
    # Dataset yapısını kontrol et
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset bulunamadı: {dataset_path}")
        return None
    
    # data.yaml dosyasını kontrol et
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    if os.path.exists(yaml_path):
        print(f"✅ data.yaml mevcut: {yaml_path}")
        return yaml_path
    
    # data.yaml yoksa oluştur
    print("📄 data.yaml oluşturuluyor...")
    
    # Sınıf isimlerini belirle (senin dataset'ine göre)
    class_names = [
        'BDC-BDR',           # ID 0
        'Caries',            # ID 1 (Çürük)
        'Fractured Teeth',   # ID 2 (Kırık)
        'Healthy Teeth',     # ID 3 (Sağlıklı)
        'Impacted teeth',    # ID 4 (Gömülü)
        'Infection'          # ID 5 (Enfeksiyon)
    ]
    
    data_yaml = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ data.yaml oluşturuldu: {yaml_path}")
    return yaml_path

def train_model(args):
    """YOLOv8 modelini eğitir"""
    
    print("=" * 60)
    print("🦷 DİŞ RÖNTGENİ NESNE TESPİTİ - MODEL EĞİTİMİ")
    print("=" * 60)
    
    # Colab ortamını ayarla
    if args.colab:
        setup_colab_environment()
    
    # Dataset'i hazırla
    if args.data_yaml is None and args.dataset_path:
        args.data_yaml = prepare_dataset_for_training(args.dataset_path)
    
    if args.data_yaml is None:
        print("❌ Eğitim için data.yaml dosyası gerekli!")
        return
    
    # Modeli yükle
    print(f"\n📦 Model yükleniyor: {args.model}")
    model = YOLO(args.model)
    
    # Eğitim parametreleri (senin kodundan)
    train_args = {
        'data': args.data_yaml,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'name': args.name,
        'patience': args.patience,
        'save': True,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print(f"\n⚙️  Eğitim Parametreleri:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")
    
    # Eğitimi başlat
    print(f"\n🚀 Eğitim başlatılıyor...")
    print(f"   Dataset: {args.data_yaml}")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    
    results = model.train(**train_args)
    
    print(f"\n✅ Eğitim tamamlandı!")
    
    # Modeli kaydet
    if args.save_to_drive and os.path.exists('/content/drive'):
        import shutil
        best_model_path = f'runs/detect/{args.name}/weights/best.pt'
        drive_model_path = f'/content/drive/MyDrive/yolov8_dental_opg_best.pt'
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, drive_model_path)
            print(f"💾 Model Google Drive'a kaydedildi: {drive_model_path}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Diş Röntgeni Model Eğitimi")
    
    # Dataset
    parser.add_argument('--data_yaml', type=str, 
                       default='/content/drive/MyDrive/Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Augmented Dataset/data.yaml',
                       help='data.yaml dosyasının yolu')
    parser.add_argument('--dataset_path', type=str,
                       default='/content/drive/MyDrive/Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Augmented Dataset',
                       help='Dataset kök dizini')
    
    # Model
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model dosyası (yolov8n.pt, yolov8s.pt)')
    
    # Eğitim parametreleri
    parser.add_argument('--epochs', type=int, default=100,
                       help='Eğitim epoch sayısı')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size (Colab için 8-16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Görüntü boyutu')
    
    # Donanım
    parser.add_argument('--device', type=str, default='0',
                       help='Cihaz (0,1,2 for GPU, "cpu" for CPU)')
    
    # Diğer
    parser.add_argument('--name', type=str, default='dental_opg_detection',
                       help='Eğitim adı')
    parser.add_argument('--patience', type=int, default=20,
                       help='Erken durdurma patience')
    
    # Colab özellikleri
    parser.add_argument('--colab', action='store_true', default=True,
                       help='Google Colab ortamında çalıştır')
    parser.add_argument('--save_to_drive', action='store_true', default=True,
                       help='Modeli Google Drive\'a kaydet')
    
    args = parser.parse_args()
    
    # Eğitimi başlat
    train_model(args)
