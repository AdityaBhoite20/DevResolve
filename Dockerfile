# Dockerfile - optional, runs the Streamlit app in a container
# Before building, run the pipeline on your computer so that indexes/final exists:
#   python src/fetch_docs.py, python src/parse_docs.py, python src/build_index.py
# Build:  docker build -t devresolve .
# Run:    docker run -p 8501:8501 --env-file .env devresolve
FROM python:3.11-slim
WORKDIR /app
# CPU-only PyTorch keeps the image much smaller than the default download
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
