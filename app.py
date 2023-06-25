from flask import Flask, request, render_template, url_for, redirect
import pickle
from markupsafe import Markup
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings
from utils.disease import disease_dic
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
app = Flask(__name__)
app.config['DEBUG'] = True
# importing pickle files
model = pickle.load(open('models/rf_pipeline.pkl', 'rb'))
ferti = pickle.load(open('models/fertname_dict.pkl', 'rb'))
loaded_model = pickle.load(open("models/RandomForest.pkl", 'rb'))
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    predictionS = disease_classes[preds[0].item()]
    # Retrieve the class label
    return predictionS


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/predict')
def pred():
    return render_template('index.html')


@app.route('/predictcrop')
def predicro():
    return render_template('home.html')


@app.route('/predict', methods=['POST',])
def predict():
    temp = int(request.form.get('temp'))
    humi = int(request.form.get('humid'))
    mois = int(request.form.get('mois'))
    soil = int(request.form.get('soil'))
    crop = int(request.form.get('crop'))
    nitro = int(request.form.get('nitro'))
    pota = int(request.form.get('pota'))
    phosp = int(request.form.get('phos'))
    input = [temp, humi, mois, soil, crop, nitro, pota, phosp]

    res = ferti[model.predict([input])[0]]

    return render_template('index.html', x=('Predicted Fertilizer is {}'.format(res)))


@app.route('/predictcrop', methods=['POST'])
def predictcrop():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(
            crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('home.html', prediction=result)


@app.route('/disease_pred')
def predco():
    return render_template('disease.html')


@ app.route('/disease')
def disease_prediction():
    title = 'Harvestify - disease Suggestion'

    return render_template('disease.html', title=title)

# render disease prediction input page


@app.route('/disease_pred', methods=['POST'])
def disease_pred():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            predictionS = predict_image(img)

            predictionS = Markup(str(disease_dic[predictionS]))
            return render_template('disease-result.html', predictionS=predictionS, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


if __name__ == "__main__":
    app.run(debug=True)
