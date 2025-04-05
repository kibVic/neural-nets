from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import json, os
from werkzeug.utils import secure_filename
from app.model_utils import load_model, predict_image

main = Blueprint('main', __name__)
model = load_model()
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main.route('/')
def index():
    return render_template('login.html')

@main.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    with open('app/users.json') as f:
        users = json.load(f)
    if username in users and users[username] == password:
        session['user'] = username
        return redirect(url_for('main.dashboard'))
    flash('Invalid credentials')
    return redirect(url_for('main.index'))

@main.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('main.index'))

@main.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('main.index'))
    return render_template('dashboard.html')


#     return render_template('upload.html')
@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('main.index'))

    prediction = None
    prediction_flour = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            # Predict the image and extract prediction results
            prediction = predict_image(model, path)

            # Get the flour with the highest probability
            predicted_flour = max(prediction, key=prediction.get)
            prediction_flour = predicted_flour.replace('_', ' ').title()  # Format for better display

            # Flash message (if needed for transient notifications)
            flash(f"Prediction: {prediction_flour} with probability: {prediction[predicted_flour]:.4f}")
            
            return redirect(url_for('main.upload'))

    return render_template('upload.html', prediction=prediction, prediction_flour=prediction_flour)
