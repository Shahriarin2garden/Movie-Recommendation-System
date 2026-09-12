# 🎬 Movie Recommendation System

> A production-ready, AI-powered movie recommendation system built with Django and advanced machine learning. Scalable from thousands to millions of movies.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-6.0-green.svg)](https://djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

![Logo Image](./assets/images-for-readme/Logo.png)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Screenshots](#-screenshots)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The Movie Recommendation System provides intelligent movie suggestions using **content-based filtering** with TF-IDF and SVD dimensionality reduction. It features a modern web interface, RESTful API, and supports datasets from 2K to 1M+ movies.


![Header Image](./assets/images-for-readme/Header.png)


### Why This Project?

- ✅ **Production Ready** - Security hardened, optimized, well-documented
- ✅ **Scalable Architecture** - Handles millions of movies efficiently
- ✅ **Modern Tech Stack** - Django 6.0, Python 3.11+, advanced ML
- ✅ **Easy to Use** - Simple installation, clear documentation
- ✅ **Flexible** - Train on 10K or 1M+ movies from any TMDB-shaped CSV

### Key Technologies

- **Backend**: Django 6.0, Python 3.11+
- **ML/Data**: scikit-learn, pandas, numpy, scipy
- **Storage**: Parquet (efficient data format)
- **Deployment**: Render, Heroku, Docker compatible

---

## 📸 Screenshots & Demo

### Demo Video

![Application Demo](./assets/demo-video/Application-Demo.gif)

### Model Loading

![Model Loading](./assets//images-for-readme/Loading.png)

### Home Page

![Home Page](./assets/images-for-readme/Homepage.png)

### Movie Search Recommendations

![Movie Recommendations](./assets/images-for-readme/Results.png)

---

## ✨ Features

### User Features
- 🔍 **Smart Search** - Real-time autocomplete with fuzzy matching
- 🎬 **AI Recommendations** - Content-based filtering with 15+ suggestions
- ⭐ **Rich Metadata** - Ratings, votes, genres, production companies
- 🔗 **External Links** - Google Search and IMDb integration
- 📱 **Responsive Design** - Works seamlessly on all devices
- ⚡ **Fast Performance** - Sub-50ms recommendation generation

### Technical Features
- 🤖 **Advanced ML** - TF-IDF + SVD dimensionality reduction
- 📊 **Scalable** - Handles 2K to 1M+ movies
- 💾 **Efficient Storage** - Parquet format with compression
- 🔧 **Configurable** - Easy model switching via `MODEL_DIR`
- 📡 **REST API** - JSON endpoints for integration
- 🔒 **Secure** - Production-ready security settings
- 📝 **Logging** - Comprehensive error tracking
- 🚀 **Deployment Ready** - `render.yaml`, `Procfile` and `Dockerfile` included

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher (numpy 2.3+ requires it)
- pip package manager
- 8GB RAM (recommended for training)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run database migrations
python manage.py migrate

# 6. Start the development server
python manage.py runserver
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8000
```

That's it — a 6,248-movie demo model ships in `demo_model/` and is found
automatically, so search works immediately with no configuration. 🎉

Want a bigger catalogue, or your own dataset? See
[Model Training](#-model-training) — it is entirely optional.

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── 📚 Documentation
│   ├── README.md                  # This file - overview and quick start
│   ├── PROJECT_GUIDE.md           # Complete technical guide
│   └── CHANGELOG.md               # Version history and changes
│
├── ⚙️ Django Application
│   ├── movie_recommendation/      # Django project settings
│   │   ├── settings.py           # Configuration
│   │   ├── urls.py               # URL routing
│   │   └── wsgi.py               # WSGI entry point
│   │
│   ├── recommender/              # Main application
│   │   ├── views.py              # Recommendation logic
│   │   ├── urls.py               # App URLs
│   │   └── templates/            # HTML templates
│   │       └── recommender/
│   │           ├── index.html    # Search page
│   │           ├── result.html   # Results page
│   │           └── error.html    # Error page
│   │
│   ├── manage.py                 # Django management script
│   └── requirements.txt          # Python dependencies
│
├── 🎓 Model Training
│   └── training/
│       ├── train.py              # Training pipeline
│       ├── infer.py              # Inference examples
│       └── guide.md              # Training documentation
│
├── 🎯 Models (Created after training)
│   ├── demo_model/               # Committed: 6,248 movies, ~3.8 MB
│   │   ├── movie_metadata.parquet    # Movie information
│   │   ├── neighbors_idx.npy         # Top-K neighbour indices
│   │   ├── neighbors_scores.npy      # Top-K neighbour scores
│   │   ├── title_to_idx.json         # Title mappings
│   │   └── config.json               # Model configuration
│   │
│   └── models/                   # git-ignored; your own trained models
│       └── ... same layout, plus *.pkl retraining artifacts
│
├── 📦 Static Files
│   └── static/
│       └── logo.ico                  # Application icon
│
├── 🚀 Deployment
│   ├── Procfile                  # Heroku configuration
│   ├── render.yaml               # Render configuration
│   ├── Dockerfile                # Container image
│   ├── .dockerignore             # Build context exclusions
│   └── .gitignore                # Git ignore rules
│
└── ⚙️ Project config
    ├── requirements.txt          # Runtime dependencies
    ├── requirements-train.txt    # Extra dependencies for training
    ├── .env.example              # Documented environment variables
    └── .github/workflows/ci.yml  # Checks and tests on every push
```

---

## 💡 Usage

### Web Interface

1. **Search for a Movie**
   - Go to `http://localhost:8000`
   - Start typing a movie name in the search box
   - Select from autocomplete suggestions or type the full name

2. **View Recommendations**
   - Click "Get Recommendations"
   - Browse 15 similar movie suggestions
   - Each card shows: rating, release date, genres, production company

3. **Explore Movies**
   - Click "Google" to search for the movie
   - Click "IMDb" to view on IMDb (if available)

### API Usage

#### Search Movies (Autocomplete)
```bash
GET /api/search/?q=matrix

Response:
{
  "movies": ["The Matrix", "The Matrix Reloaded", "The Matrix Revolutions"],
  "count": 3
}
```

#### Health Check
```bash
GET /api/health/

Response:
{
  "status": "healthy",
  "movies_loaded": 100000,
  "model_dir": "./models",
  "model_loaded": true
}
```

---

## 🎓 Model Training

### Training a Model

**Training is optional.** The committed `demo_model/` (6,248 movies with 500+
votes) covers most well-known films. Train your own for a larger catalogue, a
different language, or your own dataset. Download the
[TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
(or any CSV with the same columns), then:

```bash
pip install -r requirements-train.txt

# Top 50,000 movies by quality score, 50 neighbours each (~20 MB output)
python training/train.py ./TMDB_movie_dataset_v11.csv -o models

# Smaller and faster: only movies with 500+ votes
python training/train.py ./TMDB_movie_dataset_v11.csv -o models -q high -m 10000

python training/train.py --help   # all options
```

The trainer stores the top **K** most similar movies per title rather than a
full N x N similarity matrix. At 50,000 movies that is roughly 20 MB instead
of 10 GB, which is what makes the free tier of most hosts viable.

### Training Options

Want to train on more movies or your own dataset? See the [**Training Guide**](training/guide.md) for:

- 📖 Complete training documentation
- 🎯 Configuration options (10K to 1M+ movies)
- ⚙️ Performance tuning guidelines
- 📊 Dataset requirements
- 🔧 Advanced features

**Using the trainer from Python:**

```python
from training.train import MovieRecommenderTrainer

trainer = MovieRecommenderTrainer(
    output_dir='./models',
    use_dimensionality_reduction=True,
    n_components=500,
    top_k=50,            # neighbours stored per movie
)

df, neighbors = trainer.train(
    'path/to/your/dataset.csv',
    quality_threshold='medium',  # low/medium/high
    max_movies=50000,
)
```

**For detailed training instructions**, see:
- 📘 [Training Guide](training/guide.md) - Complete training documentation
- 📘 [PROJECT_GUIDE.md](PROJECT_GUIDE.md#-model-training) - Training setup and configurations

---

## 📡 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with search interface |
| `/` | POST | Submit movie search and get recommendations |
| `/api/search/` | GET | Search movies (autocomplete) |
| `/api/model-status/` | GET | Model loading progress |
| `/api/health/` | GET | Health check endpoint |

### Search Movies

**Request:**
```http
GET /api/search/?q=inception
```

**Response:**
```json
{
  "movies": ["Inception", "Inception: The Cobol Job"],
  "count": 2
}
```

### Health Check

**Request:**
```http
GET /api/health/
```

**Response:**
```json
{
  "status": "healthy",
  "movies_loaded": 100000,
  "model_dir": "./models",
  "model_loaded": true
}
```

For complete API documentation, see [PROJECT_GUIDE.md - API Reference](PROJECT_GUIDE.md#-api-reference)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (optional for development):

```env
# Django Settings
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Model Configuration
MODEL_DIR=./models

# Database (optional - defaults to SQLite)
# DATABASE_URL=postgresql://user:password@localhost/dbname

# Deployment
# RENDER_EXTERNAL_HOSTNAME=your-app.onrender.com
```

### Using Different Models

To switch between models, set the `MODEL_DIR` environment variable:

```bash
# Use your trained model (the default)
export MODEL_DIR=models

# Use absolute path
export MODEL_DIR=/path/to/your/models
```

For detailed configuration options, see [PROJECT_GUIDE.md - Configuration](PROJECT_GUIDE.md#-configuration)

---

## 📚 Documentation

### Main Documentation

- **[README.md](README.md)** (this file) - Overview, quick start, basic usage
- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Complete technical guide
  - Installation
  - Model training
  - Configuration
  - Development
  - Deployment
  - API reference
  - Troubleshooting
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### Training Documentation

- **[training/guide.md](training/guide.md)** - Complete model training guide
  - Dataset requirements
  - Training configurations
  - Performance tuning
  - Advanced features

### Quick Links

| Topic | Documentation |
|-------|---------------|
| Installation | [Quick Start](#-quick-start) or [PROJECT_GUIDE.md](PROJECT_GUIDE.md#-installation) |
| Model Training | [training/guide.md](training/guide.md) |
| Deployment | [PROJECT_GUIDE.md - Deployment](PROJECT_GUIDE.md#-deployment) |
| API Reference | [API Reference](#-api-reference) or [PROJECT_GUIDE.md](PROJECT_GUIDE.md#-api-reference) |
| Troubleshooting | [PROJECT_GUIDE.md - Troubleshooting](PROJECT_GUIDE.md#-troubleshooting) |
| Configuration | [Configuration](#-configuration) or [PROJECT_GUIDE.md](PROJECT_GUIDE.md#-configuration) |

---

## 🚀 Deployment

> **Deployments work as-is.** The committed `demo_model/` is found
> automatically, so a deploy from a fresh clone serves recommendations without
> any model configuration. The options below are only for shipping a *larger*
> model than the demo.

### Shipping a bigger model (optional)

**Option A — replace the demo model (simplest).**
Train into `demo_model/` (or any directory that is not git-ignored) and commit:

```bash
pip install -r requirements-train.txt
python training/train.py ./TMDB_movie_dataset_v11.csv -o demo_model -q high -m 10000
git add demo_model && git commit -m "Add demo model"
```

That is all — `demo_model/` is one of the locations checked automatically, so
you can leave `MODEL_DIR` unset (or set it explicitly if you prefer).

Two things keep this small: `models/` is git-ignored on purpose so full-size
models never enter history, while `demo_model/` is not; and `*.pkl` is ignored
repository-wide, so the retraining artifacts (an SVD pickle is easily 10× the
rest of the model combined) are skipped by `git add` automatically. Nothing
loaded at serving time is a pickle.

**Option B — fetch it at build time (keeps the repo small).**
Train a model, package it, and upload `model.tar.gz` as a GitHub Release asset:

```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o models
tar czf model.tar.gz -C models --exclude='*.pkl' .   # paths relative; no pickles
```

Set `MODEL_URL` to the asset URL. `render.yaml`'s build step downloads and
unpacks it into `MODEL_DIR`. This is build-time only — the app never fetches
anything at runtime.

### Render

1. Push to GitHub and connect the repository to [Render](https://render.com)
2. Render picks up `render.yaml` automatically
3. Set `ALLOWED_HOSTS` (your custom domain, if any) and `MODEL_URL` (option B)
4. Deploy. `SECRET_KEY` is generated for you.

### Heroku

Uses the `Procfile`. Commit a model (option A) and set config vars:

```bash
heroku config:set DEBUG=False
heroku config:set MODEL_DIR=demo_model
heroku config:set ALLOWED_HOSTS=your-app.herokuapp.com
heroku config:set SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(50))')"
```

### Docker

The image bakes in whatever is in `MODEL_DIR` at build time, so train first:

```bash
python training/train.py ./TMDB_movie_dataset_v11.csv -o models
docker build -t movie-recommender .
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(50))')"
docker run --rm -p 8000:8000 \
  -e SECRET_KEY="$SECRET_KEY" \
  -e ALLOWED_HOSTS=localhost,127.0.0.1 \
  -e SECURE_SSL_REDIRECT=False \
  movie-recommender
```

`SECURE_SSL_REDIRECT=False` is required when nothing is terminating TLS in
front of the container; otherwise every request is redirected to `https://`
and loops.

### Deployment checklist

- [ ] `SECRET_KEY` set (the app refuses to start with the dev key when `DEBUG=False`)
- [ ] `DEBUG=False`
- [ ] `ALLOWED_HOSTS` includes your domain
- [ ] A model is reachable — `curl https://your-app/api/health/` returns `"status": "healthy"`

For more detail, see [PROJECT_GUIDE.md - Deployment](PROJECT_GUIDE.md#-deployment)

---

## 🤝 Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup,
the layout of the code, and the handful of non-obvious things worth knowing
before changing the model pipeline.

The short version:

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver        # a demo model ships, so this just works

# before opening a PR -- CI runs the same three
python manage.py check
python manage.py test
ruff check .
```

Style is configured in `pyproject.toml`; `ruff check . --fix` handles the
mechanical parts. Please add a test with a bug fix and note the change under
`[Unreleased]` in `CHANGELOG.md`.

---

## 🎞️ Data & attribution

Movie metadata and poster images come from [TMDB](https://www.themoviedb.org/).

> This product uses data from TMDB but is not endorsed or certified by TMDB.

- The committed `demo_model/` is derived from the
  [TMDB Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
  and contains titles, overviews, ratings and poster paths originating from TMDB.
- Poster images are loaded directly from `image.tmdb.org` at render time; none
  are redistributed in this repository.
- If you extend this project to call the TMDB API directly, their terms require
  the wording *"This product uses the TMDB API but is not endorsed or certified
  by TMDB"* together with the TMDB logo. See
  [TMDB's terms of use](https://www.themoviedb.org/api-terms-of-use).

The MIT licence below covers this project's own source code, not the movie data.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

Need help? Here are your options:

- 📖 **Documentation**: Check [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for detailed guides
- 🎓 **Training Help**: See [training/guide.md](training/guide.md) for model training
- 🐛 **Issues**: [Open an issue](https://github.com/yourusername/movie-recommendation-system/issues) on GitHub
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/movie-recommendation-system/discussions)

---

## 🎯 Roadmap

### Version 2.1 (Planned)
- [ ] User authentication system
- [ ] Personal watchlists
- [ ] Movie rating system
- [ ] Advanced filtering (multiple genres, year ranges)
- [ ] Recommendation history

### Version 2.2 (Planned)
- [ ] Collaborative filtering
- [ ] Social features (sharing, comments)
- [ ] Movie reviews
- [ ] Advanced analytics dashboard

### Version 3.0 (Long-term)
- [ ] Mobile applications (iOS/Android)
- [ ] Real-time recommendations
- [ ] Streaming service integration
- [ ] Enhanced ML models (hybrid recommendations)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Recommendation Time | < 50ms |
| Search Response | < 100ms |
| Page Load | < 200ms |
| Memory Usage | ~200MB (100K movies) |
| Concurrent Users | 1000+ |
| Model Size | 180MB (100K movies) |

---

## 🙏 Acknowledgments

- Movie data from TMDB and IMDb
- Built with Django, scikit-learn, pandas
- UI inspired by modern design principles
- Community contributions and feedback

---

<div align="center">

**Made with ❤️ for movie lovers and developers**

[⭐ Star this repo](https://github.com/yourusername/movie-recommendation-system) •
[🐛 Report Bug](https://github.com/yourusername/movie-recommendation-system/issues) •
[💡 Request Feature](https://github.com/yourusername/movie-recommendation-system/issues)

</div>
