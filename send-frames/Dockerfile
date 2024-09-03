# Use a base Python image
FROM python:3.10-slim as base

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add a 10-second delay before starting the application
ENTRYPOINT ["sh", "-c", "sleep 10 && python main.py"]
