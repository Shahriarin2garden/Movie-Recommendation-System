# 🐳 Movie Recommendation System — Docker & Local-Run Journey

> Verified walkthrough: run the project **locally**, then run it
> **entirely on Docker** — without modifying application code.
> Written as a reference so you can reproduce everything, line by line,
> live.

---

## 1. Local run (the "before Docker" baseline)

Django 6 web app backed by pre-computed recommendation artifacts. Training is
**optional**; a committed 6,248-movie demo model makes a fresh clone work
immediately.

```bash
python3 -m venv .venv                          # Python needs 3.11+
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python manage.py check               # System check identified no issues
.venv/bin/python manage.py migrate --noinput   # create auth/session tables
.venv/bin/python manage.py runserver           # http://localhost:8000
```

Verified:

```
GET /                         -> 200 (home page)
GET /api/health/              -> {"status":"healthy","movies_loaded":6248,...}
GET /api/search/?q=inception  -> {"movies":["Inception"],"count":1}
POST / movie_name=The Matrix  -> 200, 15 recommendation cards
.venv/bin/python manage.py test                # Ran 58 tests ... OK
.venv/bin/ruff check .                         # All checks passed!
```

Training (optional, needs a TMDB-shaped CSV):

```bash
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python training/train.py TMDB_movie_dataset_v11.csv -o models -q high -m 10000
```

The model loads in a background thread; `/api/health/` reports
`"status":"starting"` until ready, then `"healthy"`.

---

## 2. Docker files (all at the repository root)

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the web app image. |
| `Dockerfile.train` | Builds the training image (`train.py` as entrypoint). |
| `docker-compose.yml` | Orchestrates `web`, plus `train`/`infer`/`test` behind profiles. |
| `.env.docker.example` | Template for the `.env` file Compose reads. |

Application code was **not** modified.

---

## 3. The web image — `Dockerfile` (line by line)

```dockerfile
FROM python:3.12-slim
```
Base image: official Python on Debian slim — small, pulls fast.

```dockerfile
ENV PYTHONUNBUFFERED=1
```
Stream Python output immediately (logs show up live in `docker compose logs`).

```dockerfile
WORKDIR /app
```
Container working directory; everything runs from here.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Install runtime deps (numpy, pandas, Django, gunicorn…). Copied *before* the
source so this heavy layer is cached and only reinstalls when requirements
change.

```dockerfile
COPY . .
RUN python manage.py collectstatic --noinput
```
Copy the app (including `demo_model/`), then collect static files. This runs
while `DEBUG` is still `True` (the default) so no `SECRET_KEY` is needed to
build the image.

```dockerfile
ENV DEBUG=False
ENV SQLITE_PATH=/data/db.sqlite3
```
Production mode. From here on the app requires `SECRET_KEY`/`ALLOWED_HOSTS`
at runtime. The SQLite database lives in `/data` so it can be a volume.

```dockerfile
RUN useradd app && mkdir -p /data && chown app:app /data
USER app
```
Create a non-root user and run as it (the `/data` dir is owned by `app` so
migrations can write the DB).

```dockerfile
EXPOSE 8000
```
Document the container port.

```dockerfile
CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn \
  movie_recommendation.wsgi:application --bind 0.0.0.0:8000 \
  --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile -"]
```
On start: run migrations (idempotent), then gunicorn. **One worker** because
the recommendation model is held per process (more workers = more memory);
**4 threads** give concurrency. `exec` makes gunicorn PID 1 so signals work.

## The training image — `Dockerfile.train`

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1        # live training progress output
WORKDIR /app
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt   # adds scikit-learn, nltk
COPY . .
ENTRYPOINT ["python", "training/train.py"]                 # CLI args follow the entrypoint
```

---

## 4. Compose — `docker-compose.yml`

| Service | Profile | Purpose |
|---------|---------|---------|
| `web` | *(default)* | `docker compose up -d` — production app on port 8000. |
| `train` | `training` | One-off training → writes artifacts into `./models`. |
| `infer` | `training` | Interactive inference / exploration (TTY). |
| `test` | `tools` | `manage.py check` + `manage.py test`. |

Key `web` settings, each explainable:

- `init: true` — Docker's tini runs as PID 1 (reaps zombies, forwards signals).
- `SECRET_KEY: "${SECRET_KEY:?...}"` — Compose **fails fast** if unset.
- `SECURE_SSL_REDIRECT: False` — no TLS proxy in front of the container;
  without this every request redirects to https and loops.
- `SQLITE_PATH: /data/db.sqlite3` + volume `db-data:/data` — DB persists.
- `healthcheck` — hits `/api/health/` with Python's `urllib` (no curl in the
  slim image); returns 200 even while the model loads, so the container isn't
  killed during the 40s `start_period`.
- `train`/`infer`/`test` are behind `profiles` so `docker compose up` starts
  only the `web` service; the others run on demand.

### 4.1 Quick start

```bash
cp .env.docker.example .env
# put a real value in SECRET_KEY (see 4.2)

docker compose up -d --build
curl http://localhost:8000/api/health/
# {"status":"healthy","movies_loaded":6248,"model_dir":"demo_model"}
```

### 4.2 Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `SECRET_KEY` | *(required)* | Compose refuses to start if unset. |
| `DEBUG` | `False` | Keep `False` in production. |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated. |
| `SECURE_SSL_REDIRECT` | `False` | `True` only behind a TLS proxy. |
| `PORT` | `8000` | Published host port → container 8000. |
| `USER_UID` / `USER_GID` | `1000` | Ownership of files written by train/infer. |

Generate `SECRET_KEY`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

### 4.3 Verified one-liners

```bash
# web
docker compose up -d --build
curl -s http://localhost:8000/api/health/                     # healthy, 6248 movies
curl -s "http://localhost:8000/api/search/?q=dark%20knight"   # 4 matches

# training (dataset as a read-only mount, output into ./models)
mkdir -p models
docker compose --profile training run --rm \
  -v "$PWD/TMDB_movie_dataset_v11.csv:/data/dataset.csv:ro" \
  train /data/dataset.csv -o models -q high -m 10000

# inference (TTY; point at ./models or override with a demo_model mount)
docker compose --profile training run --rm \
  -v "$PWD/demo_model:/app/models:ro" infer

# tests (what CI runs)
docker compose --profile tools run --rm test     # Ran 58 tests ... OK
```

Train into `./models`, then `docker compose up -d --build web` bakes the model
into the image (`.dockerignore` deliberately does **not** exclude `models/`).
To swap a model without rebuilding, uncomment a `./models:/app/models:ro`
volume in the `web` service.

---

## 5. Security posture

Kept simple, each rule is a single, explainable line:

- Non-root user (`RUN useradd app` + `USER app`).
- `init: true` (tini as PID 1).
- Secrets never baked in: `.env` is excluded by `.dockerignore`; `SECRET_KEY`
  is injected at runtime and Compose fails fast without it.
- `SECURE_SSL_REDIRECT=False` default prevents HTTPS redirect loops.
- Container healthcheck wired to the app's own `/api/health/`.

---

## 6. Gotchas worth remembering

1. **`SECRET_KEY` is mandatory.** With `DEBUG=False` the app raises
   `ImproperlyConfigured` on the dev key; Compose enforces it at startup.
2. **`collectstatic` must run before `DEBUG=False`.** Reversing the two lines
   breaks the build (no `SECRET_KEY` yet, and DEBUG is already production).
3. **One worker only.** Raising `--workers` multiplies the in-memory model by
   the worker count.
4. **`.dockerignore` vs `.gitignore`:** Docker ignores `.gitignore`. `models/`
   is not excluded (so a trained model is baked in); `demo_model/` is the
   committed fallback; `*.pkl` / `*.csv` stay out of the image.
5. **Deliberate fallback:** if no model is present anywhere, `/` still starts
   and explains what is missing; `/api/health/` returns 503.
6. **Service without building:** `infer` and `test` reuse existing images
   (`movie-recommender-train`, `movie-recommender`) — only `web` and `train`
   have build contexts.
7. **`training/infer.py` pandas pitfall:** `get_top_rated` calls `nlargest`
   after a filter that, when it empties the frame, makes pandas 3 drop the
   columns (`KeyError: 'vote_average'`). Only hits tiny synthetic datasets
   where zero movies pass the hardcoded `min_votes`; real datasets are fine
   (verified against the demo model). Pre-existing codebase quirk, not fixed
   because application code is out of scope.

---

## 7. Common workflows

```bash
docker compose up -d --build web      # rebuild + restart web
docker compose logs -f web            # tail logs
docker compose down                   # stop (add -v to drop the db volume)

rm -rf models && docker compose up -d --build web   # back to the demo model
```

## 8. Container layout

```
/app            source + collected static (read-only at runtime)
/app/demo_model committed fallback model
/app/models     optional trained model (baked in at build)
/data           SQLite database (bind/named volume)
```