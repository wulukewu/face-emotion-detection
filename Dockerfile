# --- Builder Stage ---
# This stage installs dependencies, copies code, and trains both models
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies needed for training (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Using --break-system-packages for compatibility with newer pip versions in Debian
RUN pip install --upgrade pip && pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy all code and data
COPY . .

# Run the training scripts to generate model files
# These will be copied to the final image
RUN python train_model.py && python train_cnn.py

# --- Final Stage ---
# This stage creates the final, smaller image for deployment
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy necessary files from the host
COPY app.py .
COPY feature_extractor.py .

# Copy the trained models from the builder stage
COPY --from=builder /app/emotion_model.joblib .
COPY --from=builder /app/feature_scaler.joblib .
COPY --from=builder /app/label_map.joblib .
COPY --from=builder /app/emotion_model_cnn.h5 .


# Expose port and run the application
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
CMD ["streamlit", "run", "app.py"]
