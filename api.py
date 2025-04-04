# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.xception import preprocess_input


# #instantiate the flask application
# app = Flask(__name__)

# # Load the model
# MODEL_PATH = 'final_model.h5'
# model = load_model(MODEL_PATH)

# # Define class labels
# CLASS_NAMES = [
#     "contaminated_maize_flour",
#     "maize_flour_grade_one",
#     "maize_flour_grade_two",
#     "sorghum_flour",
#     "yellow_flour"
# ]

# #prepares the image for prediction
# def prepare_image(image, target_size=(299, 299)):
#     try:
#         img = load_img(image, target_size=target_size)
#         img = img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = preprocess_input(img)
#         return img
#     except Exception as e:
#         raise ValueError(f"Image preprocessing failed: {e}")

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     image = request.files['image']

#     try:
#         img_array = prepare_image(image)
#         preds = model.predict(img_array)[0]
#         result = dict(zip(CLASS_NAMES, preds.tolist()))
#         return jsonify({'prediction': result})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/')
# def home():
#     return 'Welcome to the Flour Classification API 🌽'

# #run the api
# if __name__ == '__main__':
#     app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from io import BytesIO
from PIL import Image

# Instantiate the Flask application
app = Flask(__name__)

# Load the model
MODEL_PATH = 'final_model.h5'
model = load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = [
    "contaminated_maize_flour",
    "maize_flour_grade_one",
    "maize_flour_grade_two",
    "sorghum_flour",
    "yellow_flour"
]

# Prepares the image for prediction
def prepare_image(image, target_size=(299, 299)):
    try:
        # Convert the file to a PIL Image
        img = Image.open(BytesIO(image.read()))
        
        # Resize the image and convert to array
        img = img.resize(target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    try:
        # Prepare the image for prediction
        img_array = prepare_image(image)
        
        # Predict with the model
        preds = model.predict(img_array)[0]
        
        # Create a dictionary of class names and their predicted probabilities
        result = dict(zip(CLASS_NAMES, preds.tolist()))
        
        # Return the result as JSON
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return 'Welcome to the Flour Classification API 🌽'

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
