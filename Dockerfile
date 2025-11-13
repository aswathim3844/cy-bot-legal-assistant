# Use an official Python runtime as a parent image
# The runtime.txt in your repo specifies python-3.11.5, so we use a matching version
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the PORT environment variable that Cloud Run expects
ENV PORT 8080

# Copy the requirements file first and install dependencies
# This leverages Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn, a production-ready web server for Python
RUN pip install gunicorn

# Copy all the files from your repository into the container's /app directory
COPY . .

# Your build.sh script is CRITICAL. It runs build_vectorstore.py.
# We must run it here to ensure the faiss_index is built inside the container image.
RUN chmod +x build.sh
RUN ./build.sh

# Expose the port that gunicorn will run on
EXPOSE 8080

# This is the command to start your app using gunicorn.
# It tells gunicorn to find the 'app' object inside your 'app.py' file.
# It binds to the $PORT variable provided by Cloud Run.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
