from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('room_cleanliness_model.h5')

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file"

        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0 

        prediction = model.predict(image)

        if prediction[0][0] > 0.5:
            result = "Messy"
        else:
            result = "Clean"

        return render_template('result.html', result=result, filepath=filepath)

if __name__ == "__main__":
    app.run(debug=True)
