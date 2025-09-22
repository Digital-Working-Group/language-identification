FROM python:3.10.12-slim-bullseye
COPY requirements/py3_10_11/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip
WORKDIR /app/
# COPY scripts/ ./scripts/
# RUN for script in ./scripts/mapping_scripts/models/*/*/make_id_mappings.py; do python3 "$script"; done