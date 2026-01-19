#!/usr/bin/env python3
"""
Diş Röntgeni Nesne Tespiti - YOLOv8 Tahmin Script'i
Yazar: Sudenaz Kabay
Görselleştirme ve analiz özellikli
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

def load_classes_from_yaml(yaml_path):
    """data.yaml dosyasından sınıf isimlerini yükle"""
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', [])
    except:
        # Varsayılan sınıf isimleri (senin dataset'ine göre)
        return [
            'BDC-BDR',           # ID 0
            'Caries',            # ID 1 (Çürük)
            'Fractured Teeth',   # ID 2 (Kırık)
            'Healthy Teeth',     # ID 3 (Sağlıklı)
            'Impacted teeth',    # ID 4 (Gömülü)
            'Infection'          # ID 5 (Enfeksiyon)
        ]

def visualize_detection(image, results, classes, save_path=None):
    """Tespit sonuçlarını görselleştir"""
    
    # Renk paleti (her sınıf için farklı renk)
    colors = [
        (255, 0, 0),    # Kırmızı
        (0, 255, 0),    # Yeşil
        (0, 0, 255),    # Mavi
        (255, 255, 0),  # Sarı
        (255, 0, 255),  # Pembe
        (0, 255, 255),  # Cyan
    ]
    
    # Orijinal görseli kopyala
    img_display = image.copy()
    
    # Bounding box'ları çiz
    detections = []
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        
        for i, box in enumerate(boxes):
            # Koordinatları al
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            
            # Sınıf adı
            class_name = classes[cls_id] if cls_id < len(classes) else f'Class_{cls_id}'
            
            # Renk seç
            color = colors[cls_id % len(colors)]
            
            # Bounding box çiz
            cv2.rectangle(img_display, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Etiket yaz
            label = f"{class_name}: {conf:.2f}"
            
            # Etiket arka planı
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(img_display,
                         (int(x1), int(y1) - text_height - 5),
                         (int(x1) + text_width, int(y1)),
                         color, -1)
            
            cv2.putText(img_display, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
            
            # Tespit bilgilerini kaydet
            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class_id': cls_id
            })
    
    # Görseli göster/kaydet
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Görsel kaydedildi: {save_path}")
    
    plt.show()
    
    return img_display, detections

def analyze_and_save_results(detections, output_dir, image_name):
    """Tespit sonuçlarını analiz et ve kaydet"""
    
    # İstatistikleri hesapla
    stats = {
        'total_detections': len(detections),
        'by_class': {},
        'confidence_stats': {
            'min': min([d['confidence'] for d in detections]) if detections else 0,
            'max': max([d['confidence'] for d in detections]) if detections else 0,
            'avg': np.mean([d['confidence'] for d in detections]) if detections else 0
        },
        'timestamp': datetime.now().isoformat(),
        'image_name': image_name
    }
    
    # Sınıf bazlı istatistikler
    for det in detections:
        class_name = det['class']
        if class_name not in stats['by_class']:
            stats['by_class'][class_name] = 0
        stats['by_class'][class_name] += 1
    
    # JSON olarak kaydet
    json_path = os.path.join(output_dir, f'{Path(image_name).stem}_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    print(f"📊 İstatistikler kaydedildi: {json_path}")
    
    # İstatistikleri ekrana yazdır
    print("\n📈 TESPİT İSTATİSTİKLERİ")
    print("=" * 40)
    print(f"Toplam tespit: {stats['total_detections']}")
    print(f"Güven aralığı: {stats['confidence_stats']['min']:.2f} - {stats['confidence_stats']['max']:.2f}")
    print(f"Ortalama güven: {stats['confidence_stats']['avg']:.2f}")
    
    if stats['by_class']:
        print("\nSınıf bazlı dağılım:")
        for class_name, count in stats['by_class'].items():
            print(f"  {class_name}: {count} tespit")
    
    return stats

def main(args):
    """Ana tahmin fonksiyonu"""
    
    print("=" * 60)
    print("🦷 DİŞ RÖNTGENİ NESNE TESPİTİ - TAHMİN")
    print("=" * 60)
    
    # Modeli yükle
    print(f"\n📦 Model yükleniyor: {args.weights}")
    model = YOLO(args.weights)
    
    # Sınıf isimlerini yükle
    classes = load_classes_from_yaml(args.classes_yaml)
    print(f"📋 {len(classes)} sınıf yüklendi")
    
    # Kaynağı kontrol et
    if not os.path.exists(args.source):
        print(f"❌ Kaynak bulunamadı: {args.source}")
        return
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Kaynak bir dosya mı yoksa klasör mü?
    if os.path.isfile(args.source):
        sources = [args.source]
    else:
        # Klasördeki tüm görseller
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        sources = []
        for ext in extensions:
            sources.extend(Path(args.source).glob(f'*{ext}'))
            sources.extend(Path(args.source).glob(f'*{ext.upper()}'))
        
        if not sources:
            print(f"❌ {args.source} klasöründe görsel bulunamadı")
            return
        
        print(f"📁 {len(sources)} görsel bulundu")
    
    # Her görsel için tahmin yap
    all_detections = []
    
    for i, source_path in enumerate(sources):
        print(f"\n{'='*50}")
        print(f"🔍 [{i+1}/{len(sources)}] {Path(source_path).name}")
        print(f"{'='*50}")
        
        # Görseli yükle
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"❌ Görsel yüklenemedi: {source_path}")
            continue
        
        # Tahmin yap
        results = model.predict(
            source=str(source_path),
            conf=args.confidence,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=False,  # Kendi kaydetme fonksiyonumuzu kullanacağız
            verbose=False
        )
        
        # Görselleştir
        output_image_name = f"pred_{Path(source_path).stem}.jpg"
        output_image_path = os.path.join(args.output_dir, output_image_name)
        
        img_with_boxes, detections = visualize_detection(
            image, results[0], classes, output_image_path
        )
        
        # İstatistikleri kaydet
        stats = analyze_and_save_results(
            detections, args.output_dir, Path(source_path).name
        )
        
        all_detections.extend(detections)
        
        # Tahmin sonuçlarını txt olarak kaydet (YOLO formatında)
        if args.save_txt and detections:
            txt_path = os.path.join(args.output_dir, f"{Path(source_path).stem}.txt")
            with open(txt_path, 'w') as f:
                for det in detections:
                    # YOLO formatı: class_id x_center y_center width height confidence
                    x1, y1, x2, y2 = det['bbox']
                    img_h, img_w = image.shape[:2]
                    
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    line = f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {det['confidence']:.6f}\n"
                    f.write(line)
            
            print(f"📝 Etiketler kaydedildi: {txt_path}")
    
    # Genel istatistikleri kaydet
    if all_detections:
        total_stats = {
            'total_images': len(sources),
            'total_detections': len(all_detections),
            'average_detections_per_image': len(all_detections) / len(sources),
            'class_distribution': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for det in all_detections:
            class_name = det['class']
            if class_name not in total_stats['class_distribution']:
                total_stats['class_distribution'][class_name] = 0
            total_stats['class_distribution'][class_name] += 1
        
        summary_path = os.path.join(args.output_dir, 'summary_report.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=4, ensure_ascii=False)
        
        print(f"\n📋 Özet rapor kaydedildi: {summary_path}")
        print(f"📁 Tüm çıktılar: {args.output_dir}")
    
    print(f"\n✅ Tahmin işlemi tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Diş Röntgeni Tahmini")
    
    # Model
    parser.add_argument('--weights', type=str, 
                       default='/content/drive/MyDrive/yolov8_dental_opg_best.pt',
                       help='Eğitilmiş model dosyası')
    
    # Kaynak
    parser.add_argument('--source', type=str, 
                       default='/content/drive/MyDrive/Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Augmented Dataset/test/images',
                       help='Tahmin kaynağı (dosya/klasör)')
    
    # Sınıf bilgileri
    parser.add_argument('--classes-yaml', type=str,
                       default='/content/drive/MyDrive/Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Augmented Dataset/data.yaml',
                       help='data.yaml dosyası (sınıf isimleri için)')
    
    # Tahmin parametreleri
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Güven eşiği')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='NMS IOU eşiği')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Görüntü boyutu')
    
    # Donanım
    parser.add_argument('--device', type=str, default='0',
                       help='Cihaz (0,1,2 for GPU, "cpu" for CPU)')
    
    # Çıktı
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Çıktı dizini')
    parser.add_argument('--save-txt', action='store_true', default=True,
                       help='Etiketleri txt olarak kaydet')
    
    args = parser.parse_args()
    
    # Tahmini başlat
    main(args)
