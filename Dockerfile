# Use official TensorFlow image as parent
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p outputs/models outputs/plots

# Default command
CMD ["python", "main.py"]
