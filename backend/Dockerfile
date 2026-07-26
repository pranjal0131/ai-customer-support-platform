FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY backend ./backend
COPY ml ./ml
COPY data/demo_tickets.json ./data/demo_tickets.json
COPY models ./models

RUN addgroup --system supportiq && adduser --system --ingroup supportiq supportiq \
    && mkdir -p /app/data && chown -R supportiq:supportiq /app/data
USER supportiq

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
