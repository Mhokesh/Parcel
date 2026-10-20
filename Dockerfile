FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home dispatch
COPY . .
RUN mkdir -p /app/runtime && chown -R dispatch:dispatch /app
USER dispatch
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=2)"
CMD ["python", "run_dashboard.py", "--host", "0.0.0.0"]
