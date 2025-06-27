FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (mirror your repo structure)
COPY Data/ Data/
COPY Scripts/ Scripts/
COPY *.json .

# Run the main application (modify if needed)
CMD ["python", "Scripts/App.py"]
