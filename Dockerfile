# Use Railway's similar base image
FROM python:3.12-slim
# Set working directory
WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway uses PORT env var)
EXPOSE 8080

# Use the same start command as Railway
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080"]