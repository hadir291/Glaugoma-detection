import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow as tf
import numpy as np

# Initialiser Flask
app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Charger le modèle
model_path = "model/glaucoma_model.h5"  # Chemin vers le modèle
model = tf.keras.models.load_model(model_path)

# Paramètres du modèle
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    """
    Préparer l'image pour la prédiction.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Normaliser les pixels
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Page principale : chargement d'image et affichage du résultat.
    """
    if request.method == 'POST':
        # Vérifier si un fichier a été envoyé
        if 'file' not in request.files:
            return "Pas de fichier sélectionné"
        file = request.files['file']
        if file.filename == '':
            return "Pas de fichier sélectionné"
        
        # Sauvegarder l'image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Prédire le résultat
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        probability = prediction[0][0]
        if probability > 0.5:
            result = f"Glaucome détecté (Probabilité : {probability:.2f})"
        else:
            result = f"Pas de glaucome (Probabilité : {1 - probability:.2f})"
        
        return render_template('index.html', result=result, image_url=file_path)
    return render_template('index.html', result=None)

if __name__ == '_main_':
    app.run(debug=True)


