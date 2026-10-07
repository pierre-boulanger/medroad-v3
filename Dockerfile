FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Ensure the package is importable
RUN pip install -e . --no-deps

EXPOSE 8001

CMD ["python", "-m", "medroad_v3.main", "webhook"]
