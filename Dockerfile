# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

#Run:
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
