FROM python:3.11.14-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless tini \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    SPARK_LOCAL_IP=127.0.0.1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 1000 analyst \
    && mkdir -p data outputs \
    && chown -R analyst:analyst /app
USER analyst
ENTRYPOINT ["tini", "--"]
CMD ["churn", "--help"]
