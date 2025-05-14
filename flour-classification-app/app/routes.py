from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import json, os
from werkzeug.utils import secure_filename
from app.model_utils import load_model, predict_image
import couchdb
from datetime import datetime

# Setup
main = Blueprint("main", __name__)
model = load_model()
UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CouchDB configuration
COUCHDB_USER = "admin"
COUCHDB_PASSWORD = "admin123"
COUCHDB_URL = f"http://{COUCHDB_USER}:{COUCHDB_PASSWORD}@localhost:5984"
COUCHDB_DB_NAME = "flour_predictions"

# Connect to CouchDB
try:
    couch = couchdb.Server(COUCHDB_URL)
    db = couch[COUCHDB_DB_NAME] if COUCHDB_DB_NAME in couch else couch.create(COUCHDB_DB_NAME)

    # Create Mango index for 'timestamp' field (needed for sorting)
    db.index(
        index={"fields": ["timestamp"]},
        name="timestamp-index",
        ddoc="timestamp-index-doc",
        type="json"
    )

except couchdb.http.Unauthorized as e:
    print(f"Unauthorized error: {e}")
except Exception as e:
    print(f"Error connecting to CouchDB: {e}")

# =========================
# ROUTES
# =========================

@main.route("/")
def index():
    return render_template("login.html")


@main.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    with open("app/users.json") as f:
        users = json.load(f)
    if username in users and users[username] == password:
        session["user"] = username
        session.setdefault("images_uploaded", 0)
        session.setdefault("predictions_made", 0)
        session.setdefault("recent_predictions", [])
        return redirect(url_for("main.dashboard"))
    flash("Invalid credentials")
    return redirect(url_for("main.index"))


@main.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main.index"))


@main.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("main.index"))
    return render_template(
        "dashboard.html",
        images_uploaded=session.get("images_uploaded", 0),
        predictions_made=session.get("predictions_made", 0),
        recent_predictions=session.get("recent_predictions", []),
        last_prediction=session.get("recent_predictions", [{}])[0].get("predicted_flour", "N/A"),
    )


@main.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            prediction = predict_image(model, path)
            predicted_flour = max(prediction, key=prediction.get)
            probability = round(prediction[predicted_flour] * 100, 2)

            session["images_uploaded"] += 1
            session["predictions_made"] += 1
            new_prediction_info = {
                "image_url": url_for("static", filename=f"uploads/{filename}"),
                "image_name": filename,
                "predicted_flour": predicted_flour,
                "probability": f"{probability:.2f}",
            }

            session["recent_predictions"].insert(0, new_prediction_info)
            session["recent_predictions"] = session["recent_predictions"][:5]
            session["last_prediction"] = predicted_flour

            db.save({
                "username": session["user"],
                "image_name": filename,
                "predicted_flour": predicted_flour,
                "probability": probability,
                "timestamp": datetime.utcnow().isoformat()
            })

            flash(f"Prediction: {predicted_flour} with probability: {probability:.2f}%")
            return redirect(url_for("main.upload"))

    return render_template("upload.html")


@main.route("/get_dashboard_data")
def get_dashboard_data():
    return jsonify({
        "images_uploaded": session.get("images_uploaded", 0),
        "predictions_made": session.get("predictions_made", 0),
        "last_prediction": session.get("last_prediction", "N/A"),
        "recent_predictions": session.get("recent_predictions", []),
    })

@main.route("/latest_prediction", methods=["GET"])
def latest_prediction():
    try:
        all_docs = [db[doc_id] for doc_id in db]
        predictions = [
            doc for doc in all_docs if "predicted_flour" in doc and "timestamp" in doc
        ]

        if not predictions:
            return jsonify({"message": "No predictions found"}), 404

        predictions.sort(key=lambda x: x["timestamp"], reverse=True)

        latest = predictions[0]
        return jsonify({
            "predicted_flour": latest["predicted_flour"],
            "timestamp": latest["timestamp"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
