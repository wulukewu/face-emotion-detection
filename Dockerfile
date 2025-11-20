# --- Build Stage ---
# This stage builds the model and saves it as .joblib files
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies needed for training
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code and data needed for training
COPY . .

# Run the training script to generate the model files
RUN python train_model.py


# --- Final Stage ---
# This stage builds the final, smaller image for deployment
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files from the builder stage
COPY --from=builder /app/app.py .
COPY --from=builder /app/feature_extractor.py .
COPY --from=builder /app/emotion_model.joblib .
COPY --from=builder /app/feature_scaler.joblib .
COPY --from=builder /app/label_map.joblib .

# Expose port and run the application
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
CMD ["streamlit", "run", "app.py"]
