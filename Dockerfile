# Étape 7 — Image du dashboard de supervision (membre 4).
#
#   docker build -t batterie-dashboard .
#   docker run -p 8501:8501 -v "$(pwd)/data:/app/data" batterie-dashboard
#   → http://localhost:8501
#
# Les données ne sont pas copiées dans l'image : on monte data/ (avec data/processed/
# produit par `python main.py`) comme volume.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 : nécessaire à XGBoost et scikit-learn (OpenMP) sur l'image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/
COPY main.py .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
