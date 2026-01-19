# Dental-Xray-Object-Detection
Diş röntgeni nesne tespit projesi - Python, OpenCV/YOLO

# 🦷 Diş Röntgeni Nesne Tespiti Projesi

## 👩‍💻 Yazar
**Sudenaz Kabay**  
Bilgisayar Mühendisliği Öğrencisi  
Kırıkkale Üniversitesi

## 📌 Proje Hakkında
Bu proje, diş röntgeni görüntülerinde çürük, kırık, sağlıklı, enfeksiyon ,gömülü gibi dişlerin otomatik tespitini amaçlayan bir makine öğrenmesi uygulamasıdır.

## 🛠️ Kullanılan Teknolojiler
- **Python 3.11.7**
- **OpenCV** - Görüntü işleme
- **PyTorch/TensorFlow** - Derin öğrenme modeli
- **YOLO/CNN** - Nesne tespit mimarisi
- **Matplotlib** - Görselleştirme

## 📁 Proje Yapısı
Dental-Xray-Object-Detection/
├── 📂 data/                          # Veri seti ve etiketler
│   ├── 📂 images/                    # Diş röntgeni görüntüleri
│   │   ├── train/                    # Eğitim görüntüleri
│   │   ├── val/                      # Doğrulama görüntüleri
│   │   └── test/                     # Test görüntüleri
│   ├── 📂 labels/                    # YOLO formatında etiketler
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── 📄 classes.txt                # Sınıf isimleri: çürük, kırık, sağlıklı, enfeksiyon, gömülü
├── 📂 notebooks/                      # Jupyter Notebook'lar
│   ├── 🦷 1_data_preprocessing.ipynb   # Veri ön işleme ve analiz
│   ├── 🦷 2_model_training.ipynb       # Model eğitimi
│   ├── 🦷 3_evaluation.ipynb          # Model değerlendirme
│   └── 🦷 4_inference.ipynb           # Tahmin ve görselleştirme
├── 📂 src/                            # Python kaynak kodları
│   ├── 📄 data_loader.py              # Veri yükleme ve augmentasyon
│   ├── 📄 model.py                    # Model mimarisi
│   ├── 📄 train.py                    # Eğitim script'i
│   ├── 📄 utils.py                    # Yardımcı fonksiyonlar
│   └── 📄 visualize.py                # Görselleştirme fonksiyonları
├── 📂 weights/                         # Eğitilmiş model ağırlıkları
│   ├── 📄 best.pt                     # En iyi model
│   ├── 📄 last.pt                     # Son model
│   └── 📄 yolov8n_dental.pt           # Önceden eğitilmiş model
├── 📂 results/                         # Çıktılar ve sonuçlar
│   ├── 📂 predictions/                # Tahmin edilen görüntüler
│   ├── 📂 graphs/                     # Performans grafikleri
│   └── 📄 metrics.json                # Model metrikleri
├── 📄 requirements.txt                # Gereksinimler
├── 📄 .gitignore                      # Git yükleme dışı dosyalar
├── 📄 train.py                        # Ana eğitim script'i
├── 📄 predict.py                      # Tahmin script'i
├── 📄 config.yaml                     # YOLO konfigürasyonu
├── 🦷 dental_xray_detection.ipynb     # TÜM İŞLEMLERİ TEK NOTEBOOK
└── 📄 README.md                       # Bu dosya
