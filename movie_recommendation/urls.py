"""Root URL configuration."""
from django.conf import settings
from django.urls import include, path
from django.views.generic.base import RedirectView

# Resolved directly from STATIC_URL rather than via the {% static %} machinery:
# the manifest static storage raises at import time if collectstatic has not
# run yet, which would take the whole site down instead of one icon.
FAVICON_URL = settings.STATIC_URL.rstrip('/') + '/logo.ico'

urlpatterns = [
    path('favicon.ico', RedirectView.as_view(url=FAVICON_URL, permanent=True)),
    path('', include('recommender.urls')),
]

# Custom error pages (used when DEBUG=False).
handler404 = 'recommender.views.handler404'
handler500 = 'recommender.views.handler500'
