import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialize Flask app and database
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://flask_user:flask_password@db:5432/flour_classification')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

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

# Image preprocessing function
def prepare_image(image, target_size=(299, 299)):
    try:
        img = load_img(image, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(120), nullable=False)
    prediction_result = db.Column(db.JSON, nullable=False)  # Store the prediction result as a JSON object
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

    def __repr__(self):
        return f"<Prediction {self.id} for User {self.user.username}>"

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Logic for user login (will use POSTGRESQL for storing and checking login data)
        username = request.form['username']
        password = request.form['password']
        # Verify login (you can expand with actual user checking logic here)
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')
    return render_template('login.html')

# Route for home/dashboard after login
@app.route('/home')
def home():
    return render_template('home.html')

# Route for image upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part', 'danger')
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        try:
            img_array = prepare_image(image_path)
            preds = model.predict(img_array)[0]
            result = dict(zip(CLASS_NAMES, preds.tolist()))
        except Exception as e:
            flash(f"Error during prediction: {e}", 'danger')
            return redirect(url_for('upload_image'))

        # Store the prediction in the database
        user = User.query.filter_by(username='test_user').first()  # Use the logged-in user
        if user:
            new_prediction = Prediction(
                user_id=user.id,
                image_filename=filename,
                prediction_result=result
            )
            db.session.add(new_prediction)
            db.session.commit()

        flash(f'Prediction: {result}', 'success')
        return render_template('upload.html', result=result, image_path=image_path)
    return render_template('upload.html')

# Route for viewing predictions
@app.route('/predictions')
def view_predictions():
    user = User.query.filter_by(username='test_user').first()  # Use the logged-in user
    if user:
        predictions = Prediction.query.filter_by(user_id=user.id).all()
        return render_template('predictions.html', predictions=predictions)
    return redirect(url_for('login'))

# Route for handling logout (optional)
@app.route('/logout')
def logout():
    return redirect(url_for('login'))

# Run app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Ensure the app is accessible from Docker
