


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import base64

# app = Flask(__name__)
# CORS(app)

# print("🚀 Backend başlatıldı")

# model = YOLO("yolov8_dental_opg_best (1).pt")
# print("✅ Model yüklendi")

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "healthy"})

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         print("📥 İstek geldi")

#         if 'file' not in request.files:
#             return jsonify({"error": "Dosya yok"})

#         file = request.files['file']
#         img_bytes = file.read()

#         np_arr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         print("🧠 Model çalışıyor...")
#         results = model(img)[0]
#         print("✅ Model çıktı verdi")

#         plotted = results.plot()
#         _, buffer = cv2.imencode('.jpg', plotted)
#         result_base64 = base64.b64encode(buffer).decode('utf-8')

#         detections = []

#         # 🔥 SAĞLAM KONTROL
#         if results.boxes is not None and len(results.boxes) > 0:

#             boxes = results.boxes.xyxy.cpu().numpy()
#             confs = results.boxes.conf.cpu().numpy()
#             classes = results.boxes.cls.cpu().numpy()

#             for i in range(len(boxes)):
#                 detections.append({
#                     "confidence": float(confs[i]),
#                     "class": int(classes[i]),
#                     "bbox": boxes[i].tolist()
#                 })

#         else:
#             print("⚠️ Hiç nesne bulunamadı")

#         return jsonify({
#             "result_image": f"data:image/jpeg;base64,{result_base64}",
#             "detections": detections,
#             "total_objects": len(detections)
#         })

#     except Exception as e:
#         print("❌ HATA:", str(e))
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)








from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

print("🚀 Backend başlatıldı")

# Model yolu - doğru dosya adını kontrol edin
model_path = "yolov8_dental_opg_best.pt"

# Eğer dosya yoksa farklı isimleri dene
if not os.path.exists(model_path):
    model_path = "yolov8_dental_opg_best (1).pt"
    if not os.path.exists(model_path):
        print("⚠️ Model bulunamadı! Lütfen model dosyasını kontrol edin.")
        model = None
    else:
        model = YOLO(model_path)
        print(f"✅ Model yüklendi: {model_path}")
else:
    model = YOLO(model_path)
    print(f"✅ Model yüklendi: {model_path}")

# Model sınıflarını yazdır
if model and hasattr(model, 'names'):
    print("\n📋 Model Sınıfları:")
    for idx, name in model.names.items():
        print(f"   {idx}: {name}")

@app.route('/health', methods=['GET'])
def health():
    if model:
        return jsonify({"status": "healthy", "model_loaded": True})
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("📥 Yeni istek geldi")
        
        # Model kontrolü
        if model is None:
            return jsonify({"error": "Model yüklenemedi"}), 500
        
        # Dosya kontrolü
        if 'file' not in request.files:
            return jsonify({"error": "Dosya bulunamadı"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Dosya seçilmedi"}), 400
        
        print(f"📁 Dosya: {file.filename}")
        
        # Görseli oku
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Görsel okunamadı"}), 400
        
        print(f"🖼️ Görsel boyutu: {img.shape}")
        
        # Model tahmini
        print("🧠 Model çalışıyor...")
        results = model(img)
        result = results[0]
        print("✅ Model çıktı verdi")
        
        # Sonuç görüntüsünü oluştur
        result_img = img.copy()
        detections = []
        
        # Bounding box'ları işle
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            print(f"📦 {len(boxes)} tespit yapıldı")
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = float(confs[i])
                class_id = int(classes[i])
                
                # Sınıf adını al
                if hasattr(model, 'names') and class_id in model.names:
                    class_name = model.names[class_id]
                else:
                    class_name = f"Sınıf_{class_id}"
                
                # Renk seç (her sınıf için farklı renk)
                colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
                color = colors[class_id % len(colors)]
                
                # Bounding box çiz
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # Etiket yaz
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(result_img, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Maske varsa işle
                mask_data = None
                if result.masks is not None and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Maske alanı hesapla
                    area = np.sum(mask_binary > 127)
                    area_percentage = round((area / (img.shape[0] * img.shape[1])) * 100, 2)
                    
                    # Maskeyi base64'e çevir
                    _, mask_buffer = cv2.imencode('.png', mask_binary)
                    mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
                    
                    mask_data = {
                        'image': f"data:image/png;base64,{mask_base64}",
                        'area_percentage': area_percentage
                    }
                    
                    # Maskeyi sonuç görüntüsüne ekle
                    overlay = result_img.copy()
                    overlay[mask_binary > 0] = color
                    result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_width": int(x2 - x1),
                    "bbox_height": int(y2 - y1),
                    "center": [int((x1 + x2)/2), int((y1 + y2)/2)],
                    "mask": mask_data
                })
        else:
            print("⚠️ Hiç nesne tespit edilmedi")
        
        # Sonuç görüntüsünü base64'e çevir
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✨ Toplam tespit: {len(detections)}")
        print("="*50 + "\n")
        
        return jsonify({
            "success": True,
            "result_image": f"data:image/jpeg;base64,{result_base64}",
            "detections": detections,
            "total_objects": len(detections)
        })
        
    except Exception as e:
        print(f"❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Flask sunucusu başlatılıyor...")
    print("📍 http://127.0.0.1:5000")
    print("🔍 Sağlık kontrolü: http://127.0.0.1:5000/health")
    print("⚠️ Sunucuyu durdurmak için CTRL+C\n")
    app.run(debug=True, host='127.0.0.1', port=5000)