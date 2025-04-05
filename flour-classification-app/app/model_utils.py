import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

def load_model():
    """
    Loads the pre-trained model from a file.
    """
    return keras.models.load_model('./final_model.h5')

def predict_image(model, image_path, input_size=299):
    """
    Makes a prediction on a single image.
    """
    try:
        # Load the image and resize it to the expected input size for the model
        img = load_img(image_path, target_size=(input_size, input_size))
        
        # Convert the image to a numpy array and add a batch dimension
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess the image (depending on the model used)
        img_array = preprocess_input(img_array)

        # Get the prediction from the model
        pred = model.predict(img_array)

        # Define the class labels (ensure this matches your model's class order)
        classes = [
            "contaminated_maize_flour",
            "maize_flour_grade_one",
            "maize_flour_grade_two",
            "sorghum_flour",
            "yellow_flour"
        ]

        # Return the prediction as a dictionary
        return dict(zip(classes, pred[0]))

    except Exception as e:
        print(f"Error while predicting image: {e}")
        return None
