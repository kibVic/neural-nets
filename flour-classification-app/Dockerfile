# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the database initialization script to create the tables
RUN python init_db.py

# Expose the port the app will run on
EXPOSE 5000

# Set the environment variable for Flask to run in production mode
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the app
CMD ["flask", "run", "--host=0.0.0.0"]
