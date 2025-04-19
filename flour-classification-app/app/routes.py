from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
)
import json, os
from werkzeug.utils import secure_filename
from app.model_utils import load_model, predict_image

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

main = Blueprint("main", __name__)
model = load_model()
UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE_URL = "postgresql://root:root@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    image_name = Column(String, nullable=False)
    predicted_flour = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create table once
Base.metadata.create_all(bind=engine)


# =========================
# ROUTES SECTION
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
        if "images_uploaded" not in session:
            session["images_uploaded"] = 0
            session["predictions_made"] = 0
            session["recent_predictions"] = []
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

    images_uploaded = session.get("images_uploaded", 0)
    predictions_made = session.get("predictions_made", 0)
    recent_predictions = session.get("recent_predictions", [])
    last_prediction = (
        recent_predictions[0]["predicted_flour"] if recent_predictions else "N/A"
    )

    return render_template(
        "dashboard.html",
        images_uploaded=images_uploaded,
        predictions_made=predictions_made,
        recent_predictions=recent_predictions,
        last_prediction=last_prediction,
    )


@main.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("main.index"))

    prediction = None
    prediction_flour = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            prediction = predict_image(model, path)
            predicted_flour = max(prediction, key=prediction.get)
            prediction_flour = predicted_flour.replace("_", " ").title()

            # Session update
            session["images_uploaded"] = session.get("images_uploaded", 0) + 1
            session["predictions_made"] = session.get("predictions_made", 0) + 1
            recent_predictions = session.get("recent_predictions", [])

            image_info = {
                "image_url": url_for("static", filename=f"uploads/{filename}"),
                "image_name": filename,
                "predicted_flour": predicted_flour,
                "probability": "{:.2f}".format(prediction[predicted_flour] * 100),
            }

            recent_predictions.insert(0, image_info)
            if len(recent_predictions) > 5:
                recent_predictions = recent_predictions[:5]

            session["recent_predictions"] = recent_predictions
            session["last_prediction"] = predicted_flour

            # Insert into PostgreSQL DB
            db = SessionLocal()
            try:
                new_prediction = Prediction(
                    username=session["user"],
                    image_name=filename,
                    predicted_flour=predicted_flour,
                    probability=round(prediction[predicted_flour] * 100, 2),
                )
                db.add(new_prediction)
                db.commit()
            finally:
                db.close()

            flash(
                f"Prediction: {predicted_flour} with probability: {prediction[predicted_flour]:.4f}"
            )
            return redirect(url_for("main.upload"))

    return render_template(
        "upload.html", prediction=prediction, prediction_flour=prediction_flour
    )


@main.route("/get_dashboard_data")
def get_dashboard_data():
    images_uploaded = session.get("images_uploaded", 0)
    predictions_made = session.get("predictions_made", 0)
    recent_predictions = session.get("recent_predictions", [])
    last_prediction = session.get("last_prediction", "N/A")

    return jsonify(
        {
            "images_uploaded": images_uploaded,
            "predictions_made": predictions_made,
            "last_prediction": last_prediction,
            "recent_predictions": recent_predictions,
        }
    )


