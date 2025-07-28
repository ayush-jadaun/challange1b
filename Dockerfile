# Use the specified base image and platform
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model download script
COPY model.py .

# Copy your application source code into the container
COPY challange2b.py .

# Download and save models during Docker build
RUN python model.py

# Create input and output directories
RUN mkdir -p /app/input /app/output

# The container will run the script directly without command line arguments
# as specified in the challenge: docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
CMD ["python", "challange2b.py"]