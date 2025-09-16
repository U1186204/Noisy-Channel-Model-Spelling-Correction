# Dockerfile

# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
# This includes your .py files and the 'data' folder
COPY . .

# Specify the command to run when the container starts
CMD ["python", "test.py"]