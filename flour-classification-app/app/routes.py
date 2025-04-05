from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import json, os
import requests  
from werkzeug.utils import secure_filename
from app.model_utils import load_model, predict_image

main = Blueprint('main', __name__)
model = load_model()
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ESP32-CAM IP address
ESP32_CAM_URL = "http://<ESP32-CAM_IP>/capture"

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
        # Initialize session data for the dashboard if it's a new session
        if 'images_uploaded' not in session:
            session['images_uploaded'] = 0
            session['predictions_made'] = 0
            session['recent_predictions'] = []
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

    # Fetch current session data for display
    images_uploaded = session.get('images_uploaded', 0)
    predictions_made = session.get('predictions_made', 0)
    recent_predictions = session.get('recent_predictions', [])
    last_prediction = recent_predictions[0]['predicted_flour'] if recent_predictions else 'N/A'

    return render_template('dashboard.html', images_uploaded=images_uploaded, 
                           predictions_made=predictions_made, recent_predictions=recent_predictions,
                           last_prediction=last_prediction)

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

            # Update session data
            images_uploaded = session.get('images_uploaded', 0) + 1
            predictions_made = session.get('predictions_made', 0) + 1
            recent_predictions = session.get('recent_predictions', [])

            image_info = {
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'image_name': filename,
                'predicted_flour': predicted_flour,
                'probability': '{:.2f}'.format(prediction[predicted_flour] * 100)  # Show probability in percentage
            }

            # Add the new prediction to the list (limit to 5 most recent)
            recent_predictions.insert(0, image_info)
            if len(recent_predictions) > 5:
                recent_predictions = recent_predictions[:5]

            # Save updated session data
            session['images_uploaded'] = images_uploaded
            session['predictions_made'] = predictions_made
            session['recent_predictions'] = recent_predictions

            # Flash message (if needed for transient notifications)
            flash(f"Prediction: {predicted_flour} with probability: {prediction[predicted_flour]:.4f}")
            
            # Update the last prediction in session
            session['last_prediction'] = predicted_flour

            return redirect(url_for('main.upload'))

    return render_template('upload.html', prediction=prediction, prediction_flour=prediction_flour)

# Capture images using ESP32-CAM
@main.route('/capture-image')
def capture_image():
    try:
        # Send a request to ESP32-CAM to capture an image
        response = requests.get(ESP32_CAM_URL)
        if response.status_code == 200:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Route for real-time dashboard data
@main.route('/get_dashboard_data')
def get_dashboard_data():
    images_uploaded = session.get('images_uploaded', 0)
    predictions_made = session.get('predictions_made', 0)
    recent_predictions = session.get('recent_predictions', [])
    last_prediction = session.get('last_prediction', 'N/A')  # Use 'last_prediction' from session

    return jsonify({
        'images_uploaded': images_uploaded,
        'predictions_made': predictions_made,
        'last_prediction': last_prediction,
        'recent_predictions': recent_predictions
    })
