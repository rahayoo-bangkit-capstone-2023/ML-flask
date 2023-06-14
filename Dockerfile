# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 8080

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Specify the command to run the Flask application
CMD ["python", "app.py"] 