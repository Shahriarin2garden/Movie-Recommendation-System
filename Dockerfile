# Movie Recommendation System
#
# The image bakes in whatever is in your MODEL_DIR at build time, so train a
# model first (see training/guide.md), then build:
#
#   python training/train.py ./TMDB_movie_dataset_v11.csv -o models
#   docker build -t movie-recommender .
#   docker run --rm -p 8000:8000 \
#       -e SECRET_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(50))')" \
#       -e ALLOWED_HOSTS=localhost,127.0.0.1 \
#       -e SECURE_SSL_REDIRECT=False \
#       movie-recommender
#
# SECURE_SSL_REDIRECT must be False when there is no TLS-terminating proxy in
# front, otherwise every request is redirected to https:// and loops.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer is cached across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Collected while DEBUG still defaults to True: with DEBUG=False the settings
# module refuses to load without a real SECRET_KEY, which would fail the build.
RUN python manage.py collectstatic --noinput

ENV DEBUG=False \
    MODEL_DIR=models \
    PORT=8000

EXPOSE 8000

# The recommendation model is held per process, so one worker with threads
# keeps memory to a single copy.
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn movie_recommendation.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 --log-file - --access-logfile -"]
