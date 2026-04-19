FROM public.ecr.aws/docker/library/python:3.10-slim

ARG CACHEBUST=1

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
