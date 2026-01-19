#!/usr/bin/env python3
"""
Diş Röntgeni Görselleştirme - YOLOv8 sonuçları için
Yazar: Sudenaz Kabay
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

CLASS_COLORS = {
    'BDC-BDR': (255, 0, 0),        # Kırmızı
    'Caries': (0, 255, 0),         # Yeşil (Çürük)
    'Fractured Teeth': (0, 0, 255), # Mavi (Kırık)
    'Healthy Teeth': (255, 255, 0), # Sarı (Sağlıklı)
    'Impacted teeth': (255, 0, 255), # Pembe (Gömülü)
    'Infection': (0, 255, 255)     # Cyan (Enfeksiyon)
}

def plot_yolo_detections(image_path: str, model, confidence_threshold: float = 0.25):
    print(f"🔍 Görsel analiz ediliyor: {Path(image_path).name}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Görsel yüklenemedi: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=False,
        verbose=False
    )
    
    plotted_result = results[0].plot()
    plotted_result_rgb = cv2.cvtColor(plotted_result, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Orijinal Görsel", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(plotted_result_rgb)
    axes[1].set_title("YOLOv8 Tespitleri", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        print(f"\n📊 TESPİT İSTATİSTİKLERİ:")
        print("-" * 40)
        
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            
            if cls_name not in class_counts:
                class_counts[cls_name] = {'count': 0, 'confidences': []}
            
            class_counts[cls_name]['count'] += 1
            class_counts[cls_name]['confidences'].append(conf)
        
        for cls_name, data in class_counts.items():
            avg_conf = np.mean(data['confidences'])
            print(f"  {cls_name}: {data['count']} nesne, ortalama güven: {avg_conf:.3f}")
    else:
        print("\n⚠️  Hiç nesne tespit edilmedi")
    
    return results

def plot_multiple_predictions(test_images_dir: str, model_path: str, num_images: int = 4):
    print("📸 BİRDEN FAZLA GÖRSEL TAHMİNİ")
    print("=" * 60)
    
    from ultralytics import YOLO
    model = YOLO(model_path)
    
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend(Path(test_images_dir).glob(f'*{ext}'))
        test_images.extend(Path(test_images_dir).glob(f'*{ext.upper()}'))
    
    if not test_images:
        print(f"❌ Test görseli bulunamadı: {test_images_dir}")
        return
    
    test_images = sorted(test_images)[:num_images]
    
    for i, img_path in enumerate(test_images):
        print(f"\n{'='*50}")
        print(f"🦷 [{i+1}/{len(test_images)}] {img_path.name}")
        print(f"{'='*50}")
        
        results = plot_yolo_detections(str(img_path), model)
        plt.pause(1)
    
    print(f"\n✅ {len(test_images)} görsel analiz edildi")

def save_detection_results(results, image_path: str, output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)
    
    output_image_path = os.path.join(output_dir, f'pred_{Path(image_path).name}')
    plotted_result = results[0].plot()
    cv2.imwrite(output_image_path, plotted_result)
    
    stats_path = os.path.join(output_dir, f'stats_{Path(image_path).stem}.txt')
    boxes = results[0].boxes
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Görsel: {Path(image_path).name}\n")
        f.write(f"Analiz Tarihi: {pd.Timestamp.now()}\n")
        f.write("=" * 40 + "\n")
        
        if boxes is not None and len(boxes) > 0:
            f.write(f"Toplam Tespit: {len(boxes)}\n\n")
            
            for i, box in enumerate(boxes, 1):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]
                
                f.write(f"Tespit {i}:\n")
                f.write(f"  Sınıf: {cls_name}\n")
                f.write(f"  Güven: {conf:.4f}\n")
                f.write("-" * 30 + "\n")
        else:
            f.write("Hiç nesne tespit edilmedi.\n")
    
    print(f"✅ Sonuçlar kaydedildi: {output_image_path}, {stats_path}")

def test_visualization():
    print("🎨 Görselleştirme Testi")
    print("=" * 50)
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite('test_image.jpg', test_image)
    
    print("✅ Test görseli oluşturuldu")
    
    if os.path.exists('test_image.jpg'):
        os.remove('test_image.jpg')
    
    print("✅ Görselleştirme testi tamamlandı")

if __name__ == "__main__":
    test_visualization()
