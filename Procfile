web: gunicorn movie_recommendation.wsgi:application --workers 1 --threads 4 --timeout 120 --log-file - --access-logfile - --log-level info
release: python manage.py migrate --noinput
