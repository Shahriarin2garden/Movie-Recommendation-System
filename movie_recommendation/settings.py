"""
Django settings for the Movie Recommendation System.

All deployment-specific values are read from the environment so the same
codebase runs locally and in production. See `.env.example` for the full list
of supported variables.
"""
import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name, default=False):
    """Read a boolean from the environment ('1', 'true', 'yes', 'on' are truthy)."""
    return os.environ.get(name, str(default)).strip().lower() in ('true', '1', 't', 'yes', 'on')


def env_list(name, default=''):
    """Read a comma-separated list from the environment, dropping blanks."""
    return [item.strip() for item in os.environ.get(name, default).split(',') if item.strip()]


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env_bool('DEBUG', True)

# SECURITY WARNING: keep the secret key used in production secret!
INSECURE_DEV_SECRET_KEY = 'django-insecure-dev-key-change-in-production-12345'
SECRET_KEY = os.environ.get('SECRET_KEY') or INSECURE_DEV_SECRET_KEY

if not DEBUG and SECRET_KEY == INSECURE_DEV_SECRET_KEY:
    raise ImproperlyConfigured(
        'SECRET_KEY must be set to a unique, secret value when DEBUG=False. '
        'Generate one with: '
        'python -m django shell -c "from django.core.management.utils import '
        'get_random_secret_key as k; print(k())"'
    )

# Hosts allowed to serve this site.
ALLOWED_HOSTS = env_list('ALLOWED_HOSTS', 'localhost,127.0.0.1,[::1]' if DEBUG else '')

# Origins trusted for CSRF-protected POSTs (required by Django behind HTTPS).
CSRF_TRUSTED_ORIGINS = env_list('CSRF_TRUSTED_ORIGINS')

# Render deployment support: the platform injects the public hostname.
RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
    CSRF_TRUSTED_ORIGINS.append('https://' + RENDER_EXTERNAL_HOSTNAME)

# Application definition
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "recommender.apps.RecommenderConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Security settings
X_FRAME_OPTIONS = 'DENY'
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = 'same-origin'
SESSION_COOKIE_HTTPONLY = True

if not DEBUG:
    # Platforms such as Render and Heroku terminate TLS at their proxy. Without
    # this header Django never sees a request as secure, and SECURE_SSL_REDIRECT
    # would bounce the client in an infinite redirect loop.
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = env_bool('SECURE_SSL_REDIRECT', True)
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', 60 * 60 * 24 * 365))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

ROOT_URLCONF = "movie_recommendation.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "movie_recommendation.wsgi.application"


# Database
# The recommender itself is stateless and file-backed; SQLite only backs
# Django's built-in session and auth tables.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get('SQLITE_PATH') or (BASE_DIR / "db.sqlite3"),
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# Caching -- backs the autocomplete endpoint so repeated prefixes are free.
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'movie-recommendation-cache',
        'OPTIONS': {'MAX_ENTRIES': 5000},
    }
}

# Logging -- console only. Log files are a poor fit for containerised platforms
# with ephemeral disks, and the previous file handler pointed at a directory
# that is not in the repo, which crashed startup before Django could report it.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'stream': sys.stdout,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG' if DEBUG else 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'recommender': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}

# Movie recommendation model configuration.
# Point MODEL_DIR at any directory produced by training/train.py. Relative
# paths resolve against BASE_DIR so the app behaves the same regardless of the
# working directory gunicorn happens to start in.
MODEL_DIR = Path(os.environ.get('MODEL_DIR') or (BASE_DIR / 'models'))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (BASE_DIR / MODEL_DIR).resolve()

# Number of recommendations returned for a search.
RECOMMENDATION_COUNT = int(os.environ.get('RECOMMENDATION_COUNT', 15))

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
