
#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import base64
import numpy as np
import cv2
import torch
from PIL import Image
from io import BytesIO
import time
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import AutoProcessor, AutoModelForImageClassification as EmotionModel
import os

app = Flask(__name__)
CORS(app)

# Initialisation des modèles et configurations
print("Initialisation des modèles d'IA...")

# 1. Modèle pour la détection de personnes (YOLO)
# Utilisation de YOLOv5 via torch hub
model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model_detection.classes = [0]  # Classe 0 = personne
model_detection.conf = 0.4  # Seuil de confiance

# 2. Modèle pour la classification de posture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
posture_model_name = "mhdrcs/human-posture-classification"
posture_extractor = AutoFeatureExtractor.from_pretrained(posture_model_name)
posture_model = AutoModelForImageClassification.from_pretrained(posture_model_name).to(device)
posture_classes = ["debout", "assis", "courbé", "allongé", "acrobatique"]

# 3. Modèle pour l'analyse d'émotion et de confiance
emotion_model_name = "dima806/facial_emotions_image_detection"
emotion_processor = AutoProcessor.from_pretrained(emotion_model_name)
emotion_model = EmotionModel.from_pretrained(emotion_model_name).to(device)
emotion_classes = ["neutre", "joie", "surprise", "tristesse", "colère", "peur", "dégoût"]

print("Modèles chargés avec succès!")

def base64_to_image(base64_string):
    """Convertir une image base64 en tableau numpy pour traitement"""
    img_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    return img

def detect_persons(image):
    """Détecter les personnes à l'aide du modèle YOLOv5"""
    results = model_detection(image)
    
    # Traiter les résultats et extraire les informations nécessaires
    detections = []
    results_data = results.pandas().xyxy[0]
    
    for idx, row in results_data.iterrows():
        if row['class'] == 0:  # Classe 0 = personne
            box = {
                'x1': int(row['xmin']),
                'y1': int(row['ymin']),
                'x2': int(row['xmax']),
                'y2': int(row['ymax']),
                'confidence': float(row['confidence'])
            }
            detections.append(box)
    
    return detections

def analyze_posture(img, box):
    """Analyser la posture d'une personne détectée"""
    # Extraction de la région de la personne
    person_img = img[box['y1']:box['y2'], box['x1']:box['x2']]
    
    # Vérifier si la boîte est valide et contient une image
    if person_img.size == 0:
        return {"posture": "inconnue", "confidence": 0.0}
    
    # Conversion en PIL et préparation pour le modèle
    person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
    
    # Prétraitement de l'image
    inputs = posture_extractor(person_pil, return_tensors="pt").to(device)
    
    # Inférence
    with torch.no_grad():
        outputs = posture_model(**inputs)
    
    # Analyse des résultats
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)
    
    # Retourner la posture prédite et le score de confiance
    posture = posture_classes[top_idx.item()] if top_idx.item() < len(posture_classes) else "inconnue"
    confidence = top_prob.item()
    
    return {"posture": posture, "confidence": confidence}

def detect_face_and_emotion(img, box):
    """Détecter le visage et analyser les émotions"""
    # Extraction de la région supérieure (visage potentiel)
    face_height = int((box['y2'] - box['y1']) * 0.4)  # Zone supérieure potentielle du visage
    face_img = img[box['y1']:box['y1'] + face_height, box['x1']:box['x2']]
    
    # Vérifier si la boîte est valide et contient une image
    if face_img.size == 0:
        return {"emotion": "inconnue", "confidence": 0.0, "assurance": 0.0}
    
    # Conversion en PIL
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    
    # Prétraitement pour le modèle d'émotion
    inputs = emotion_processor(face_pil, return_tensors="pt").to(device)
    
    # Inférence
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    
    # Analyse des résultats
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_idx = torch.max(probs, dim=-1)
    
    # Calcul du niveau d'assurance basé sur l'émotion et la posture
    emotion_code = top_idx.item() if top_idx.item() < len(emotion_classes) else 0
    assurance = 0.0
    
    # Logique simple pour calculer l'assurance:
    # Joie et Neutre suggèrent généralement plus d'assurance
    if emotion_code == 1 or emotion_code == 0:  # Joie ou Neutre
        assurance = 0.7 + (top_prob.item() * 0.3)
    elif emotion_code == 2:  # Surprise
        assurance = 0.5 + (top_prob.item() * 0.3)
    else:  # Émotions négatives
        assurance = 0.3 * top_prob.item()
    
    emotion = emotion_classes[emotion_code] if emotion_code < len(emotion_classes) else "inconnue"
    confidence = top_prob.item()
    
    return {"emotion": emotion, "confidence": confidence, "assurance": assurance}

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    """Point d'entrée API pour l'analyse de l'image"""
    start_time = time.time()
    
    if 'image' not in request.json:
        return jsonify({"error": "Aucune image fournie"}), 400
    
    # Obtenir l'image depuis les données POST
    image_data = request.json['image']
    img = base64_to_image(image_data)
    
    # 1. Détecter les personnes
    person_boxes = detect_persons(img)
    
    # Analyser chaque personne détectée
    analyzed_persons = []
    for i, box in enumerate(person_boxes):
        # 2. Analyser la posture
        posture_data = analyze_posture(img, box)
        
        # 3. Analyser l'émotion et le taux d'assurance
        emotion_data = detect_face_and_emotion(img, box)
        
        # Assembler les données pour cette personne
        person_data = {
            "id": i + 1,
            "box": box,
            "posture": posture_data["posture"],
            "posture_confidence": posture_data["confidence"],
            "emotion": emotion_data["emotion"],
            "emotion_confidence": emotion_data["confidence"],
            "assurance": emotion_data["assurance"]
        }
        analyzed_persons.append(person_data)
    
    # 4. Préparer la réponse avec toutes les informations
    response = {
        "total_persons": len(person_boxes),
        "persons": analyzed_persons,
        "processing_time": time.time() - start_time
    }
    
    return jsonify(response)

@app.route('/')
def home():
    return "API d'Analyse de Personnes - Active"

if __name__ == '__main__':
    # Démarrage du serveur Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
