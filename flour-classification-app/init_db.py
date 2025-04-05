# init_db.py
from app import db

# Create the database tables
db.create_all()

print("Database tables created successfully.")
