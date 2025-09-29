FROM python:3.10.12-slim-bullseye
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip
WORKDIR /app/