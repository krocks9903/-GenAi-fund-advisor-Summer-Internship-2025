# Use official Python slim image as base
FROM python:3.11-slim

# Install system dependencies needed for Chrome, Selenium, and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libappindicator3-1 \
    libasound2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    unzip \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome headless use by Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_DRIVER=/usr/lib/chromium/chromedriver

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code, including your FAISS index folder
COPY . .

# OPTIONAL: confirm index file is copied (for debugging)
# RUN ls -R /app

# Expose port if using Streamlit (default 8501)
EXPOSE 8501

# Run your Streamlit app
CMD ["streamlit", "run", "Scripts/App.py", "--server.port=8501", "--server.address=0.0.0.0"]
