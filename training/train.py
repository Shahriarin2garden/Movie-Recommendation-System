"""
Advanced Movie Recommendation System - Training Pipeline
Optimized for TMDB Movies Dataset 2023 (930K+ movies)
"""

import sys

# Windows consoles default to a legacy code page (cp1252) that cannot encode
# the emoji used in this script's progress output, which crashes the run.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import pickle
import warnings
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


class MovieRecommenderTrainer:
    def __init__(self, output_dir='./models', use_dimensionality_reduction=True,
                 n_components=500, top_k=50, chunk_size=1024):
        """
        Initialize the trainer with advanced configurations

        Args:
            output_dir: Directory to save trained models
            use_dimensionality_reduction: Use SVD to reduce memory footprint
            n_components: Number of latent features for SVD
            top_k: Number of nearest neighbours stored per movie
            chunk_size: Rows compared at a time when computing neighbours
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_svd = use_dimensionality_reduction
        self.n_components = n_components
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.stemmer = SnowballStemmer('english')

    def load_data(self, data_path):
        """
        Load TMDB dataset from single CSV file

        Args:
            data_path: Path to TMDB_movie_dataset_v11.csv

        Returns:
            DataFrame with movie data
        """
        print("Loading TMDB dataset...")

        # Handle both file path and directory path
        if Path(data_path).is_file():
            df = pd.read_csv(data_path, low_memory=False)
        else:
            # Assume it's a directory
            csv_path = Path(data_path) / 'TMDB_movie_dataset_v11.csv'
            df = pd.read_csv(csv_path, low_memory=False)

        print(f"Loaded {len(df)} movies")
        print(f"Columns: {df.columns.tolist()}")

        return df

    def parse_json_column(self, col_data, key='name'):
        """
        Parse JSON-like string columns (genres, keywords, production_companies)
        Handles both string representation and actual lists
        """
        if pd.isna(col_data) or col_data == '' or col_data == '[]':
            return []

        try:
            # Try literal_eval first
            parsed = literal_eval(col_data) if isinstance(col_data, str) else col_data

            if isinstance(parsed, list):
                # Extract the specified key from each dict
                return [item[key] for item in parsed if isinstance(item, dict) and key in item]
            return []
        except (ValueError, SyntaxError, TypeError, KeyError):
            # Fallback: split by comma if it's a simple comma-separated string
            if isinstance(col_data, str):
                return [item.strip() for item in col_data.split(',') if item.strip()]
            return []

    def extract_director_from_companies(self, companies_data):
        """
        Extract primary production company as a proxy for director
        (TMDB dataset doesn't have separate crew/director info)
        """
        companies = self.parse_json_column(companies_data)
        return companies[0] if companies else None

    def clean_and_engineer_features(self, df, quality_threshold='medium'):
        """
        Advanced feature engineering pipeline for TMDB dataset

        Args:
            df: Input DataFrame
            quality_threshold: 'low', 'medium', or 'high' - filters by vote_count

        Returns:
            Processed DataFrame
        """
        print("Engineering features...")

        # Filter by quality threshold
        thresholds = {
            'low': 5,      # 5+ votes
            'medium': 50,  # 50+ votes (recommended)
            'high': 500    # 500+ votes (high quality only)
        }
        min_votes = thresholds.get(quality_threshold, 50)
        df = df[df['vote_count'] >= min_votes].copy()
        print(f"Filtered to {len(df)} movies with {min_votes}+ votes")

        # Filter only released movies
        df = df[df['status'] == 'Released'].copy()

        # Parse JSON columns
        print("Parsing genres, keywords, and production companies...")
        df['genres'] = df['genres'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['keywords'] = df['keywords'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['companies'] = df['production_companies'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['countries'] = df['production_countries'].apply(lambda x: self.parse_json_column(x, 'name'))

        # Extract primary production company as director proxy
        df['primary_company'] = df['companies'].apply(lambda x: x[0] if x else None)

        # Process overview (plot summary)
        df['overview_clean'] = df['overview'].fillna('').astype(str)
        df['overview_words'] = df['overview_clean'].apply(
            lambda x: [word.lower() for word in x.split()[:50]]  # First 50 words
        )

        # Process tagline
        df['tagline_clean'] = df['tagline'].fillna('').astype(str)
        df['tagline_words'] = df['tagline_clean'].apply(
            lambda x: [word.lower() for word in x.split()]
        )

        # Clean and stem keywords
        df['keywords'] = df['keywords'].apply(
            lambda x: [self.stemmer.stem(kw.lower().replace(" ", "")) for kw in x[:15]]  # Top 15 keywords
        )

        # Clean genres
        df['genres'] = df['genres'].apply(
            lambda x: [genre.lower().replace(" ", "") for genre in x]
        )

        # Clean companies (top 3, with weight)
        df['companies_weighted'] = df['companies'].apply(
            lambda x: [x[0].lower().replace(" ", "")] * 2 if x and len(x) > 0 else []  # Weight first company
        )
        df['companies_clean'] = df['companies'].apply(
            lambda x: [comp.lower().replace(" ", "") for comp in x[:3]]
        )

        # Clean countries
        df['countries_clean'] = df['countries'].apply(
            lambda x: [country.lower().replace(" ", "") for country in x[:2]]
        )

        # Create comprehensive soup feature
        df['soup'] = (
            df['keywords'] +
            df['genres'] * 2 +  # Weight genres more
            df['companies_weighted'] +
            df['companies_clean'] +
            df['countries_clean'] +
            df['overview_words'] +
            df['tagline_words']
        )
        df['soup'] = df['soup'].apply(lambda x: ' '.join(x) if x else '')

        # Filter valid entries
        df = df[df['soup'].str.len() > 20].copy()
        df = df.dropna(subset=['title'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')

        # Sort by popularity (combination of vote_average and vote_count)
        df['quality_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df = df.sort_values('quality_score', ascending=False)

        if 'tconst' in df.columns and 'imdb_id' not in df.columns:
          df['imdb_id'] = df['tconst']

        df = df.reset_index(drop=True)

        print(f"Processed {len(df)} valid movies")
        return df

    def build_tfidf_matrix(self, df):
        """Build TF-IDF matrix with optimized parameters"""
        print("Building TF-IDF matrix...")

        # Adjust max_features based on dataset size
        n_movies = len(df)
        if n_movies < 10000:
            max_features = 10000
        elif n_movies < 100000:
            max_features = 15000
        else:
            max_features = 20000

        print(f"Using max_features={max_features} for {n_movies} movies")

        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=3,  # Increased for larger dataset
            max_df=0.7,  # More aggressive filtering
            stop_words='english',
            max_features=max_features,
            sublinear_tf=True  # Use log scaling
        )

        tfidf_matrix = tfidf.fit_transform(df['soup'])

        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
        print(f"Matrix sparsity: {sparsity:.2f}%")

        return tfidf_matrix, tfidf

    def build_features(self, tfidf_matrix):
        """Reduce the TF-IDF matrix to dense latent features when SVD is on."""
        if not (self.use_svd and tfidf_matrix.shape[0] > 1000):
            return tfidf_matrix, None

        print(f"Applying SVD dimensionality reduction to {self.n_components} components...")
        n_components = min(
            self.n_components,
            tfidf_matrix.shape[0] - 1,
            tfidf_matrix.shape[1] - 1,
        )
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(tfidf_matrix).astype(np.float32)
        print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
        print(f"Reduced matrix shape: {reduced.shape}")
        return reduced, svd

    def compute_neighbors(self, features):
        """Compute the top-K most similar movies for every movie.

        Storing K neighbours per movie instead of the full N x N similarity
        matrix is what makes large catalogues practical: at 50,000 movies the
        full matrix is ~10 GB, while K=50 neighbours is about 20 MB. Similarity
        is computed in row chunks so peak memory stays bounded regardless of N.
        """
        n_movies = features.shape[0]
        k = int(min(self.top_k, n_movies - 1))
        if k < 1:
            raise ValueError("Need at least two movies to compute neighbours")

        print(f"Computing top-{k} neighbours for {n_movies:,} movies "
              f"(chunk size {self.chunk_size})...")

        neighbor_idx = np.empty((n_movies, k), dtype=np.int32)
        neighbor_scores = np.empty((n_movies, k), dtype=np.float32)
        n_chunks = (n_movies + self.chunk_size - 1) // self.chunk_size

        for chunk_no, start in enumerate(range(0, n_movies, self.chunk_size), start=1):
            end = min(start + self.chunk_size, n_movies)
            sims = cosine_similarity(features[start:end], features).astype(np.float32)

            # A movie is always its own nearest neighbour; drop it.
            rows = np.arange(end - start)
            sims[rows, np.arange(start, end)] = -np.inf

            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(sims, top, axis=1)
            order = np.argsort(-top_scores, axis=1)

            neighbor_idx[start:end] = np.take_along_axis(top, order, axis=1)
            neighbor_scores[start:end] = np.take_along_axis(top_scores, order, axis=1)

            if chunk_no % 5 == 0 or chunk_no == n_chunks:
                print(f"  processed {chunk_no}/{n_chunks} chunks")

        size_mb = (neighbor_idx.nbytes + neighbor_scores.nbytes) / 1024 ** 2
        print(f"Neighbour data: {neighbor_idx.shape} ({size_mb:.1f} MB)")
        return neighbor_idx, neighbor_scores

    def compute_similarity_matrix(self, tfidf_matrix):
        """Compute the full similarity matrix (legacy format).

        Kept for backwards compatibility. Prefer compute_neighbors: this
        allocates an N x N array and will exhaust memory on large catalogues.
        """
        if self.use_svd and tfidf_matrix.shape[0] > 1000:
            print(f"Applying SVD dimensionality reduction to {self.n_components} components...")

            # Adjust components based on matrix size
            n_components = min(
                self.n_components,
                tfidf_matrix.shape[0] - 1,
                tfidf_matrix.shape[1] - 1
            )

            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)

            explained_var = svd.explained_variance_ratio_.sum()
            print(f"Explained variance ratio: {explained_var:.3f}")
            print(f"Reduced matrix shape: {reduced_matrix.shape}")

            # For very large datasets, compute similarity in chunks
            if reduced_matrix.shape[0] > 50000:
                print("Computing similarity in chunks for large dataset...")
                chunk_size = 10000
                n_chunks = (reduced_matrix.shape[0] + chunk_size - 1) // chunk_size

                similarity_matrix = np.zeros((reduced_matrix.shape[0], reduced_matrix.shape[0]), dtype=np.float32)

                for i in range(n_chunks):
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, reduced_matrix.shape[0])

                    chunk_sim = cosine_similarity(
                        reduced_matrix[start_i:end_i],
                        reduced_matrix
                    )
                    similarity_matrix[start_i:end_i, :] = chunk_sim

                    if (i + 1) % 5 == 0:
                        print(f"Processed {i+1}/{n_chunks} chunks")
            else:
                print("Computing cosine similarity...")
                similarity_matrix = cosine_similarity(reduced_matrix)

            return similarity_matrix.astype(np.float32), svd
        else:
            print("Computing cosine similarity (no dimensionality reduction)...")
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return similarity_matrix.astype(np.float32), None

    def save_model(self, df, tfidf_vectorizer, svd_model=None,
                   neighbors=None, similarity_matrix=None):
        """Save all model artifacts efficiently"""
        print("Saving model artifacts...")

        # Save metadata DataFrame (essential columns only). Columns missing
        # from a custom dataset are filled in rather than raising a KeyError.
        wanted = [
            'id', 'title', 'release_date', 'primary_company',
            'genres', 'vote_average', 'vote_count', 'popularity',
            'overview', 'imdb_id', 'poster_path'
        ]
        for column in wanted:
            if column not in df.columns:
                print(f"  note: column '{column}' missing from dataset, storing as empty")
                df[column] = None

        metadata_df = df[wanted].copy()
        metadata_df.to_parquet(
            self.output_dir / 'movie_metadata.parquet',
            compression='gzip',
            index=True
        )

        matrix_shape = None
        if neighbors is not None:
            neighbor_idx, neighbor_scores = neighbors
            np.save(self.output_dir / 'neighbors_idx.npy', neighbor_idx)
            np.save(self.output_dir / 'neighbors_scores.npy', neighbor_scores)
            matrix_shape = list(neighbor_idx.shape)
            print(f"Saved top-{neighbor_idx.shape[1]} neighbours "
                  f"({(neighbor_idx.nbytes + neighbor_scores.nbytes) / 1024**2:.1f} MB)")
        elif similarity_matrix is not None:
            print("Saving full similarity matrix (legacy format)...")
            matrix_shape = list(similarity_matrix.shape)
            density = np.count_nonzero(similarity_matrix) / similarity_matrix.size
            # csr only saves space on a genuinely sparse matrix; a dense one
            # stored as csr is roughly 50% larger than the plain array.
            if density < 0.3:
                sparse_sim = csr_matrix(similarity_matrix)
                save_npz(self.output_dir / 'similarity_matrix.npz', sparse_sim)
                print(f"Saved as sparse matrix ({sparse_sim.data.nbytes / 1024**2:.1f} MB)")
            else:
                np.save(self.output_dir / 'similarity_matrix.npy', similarity_matrix)
                print(f"Saved as dense matrix ({similarity_matrix.nbytes / 1024**2:.1f} MB)")

        # Save title to index mapping
        title_to_idx = pd.Series(df.index, index=df['title']).to_dict()
        with open(self.output_dir / 'title_to_idx.json', 'w', encoding='utf-8') as f:
            json.dump(title_to_idx, f, ensure_ascii=False)

        # Save TF-IDF vectorizer
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

        # Save SVD model if used
        if svd_model is not None:
            with open(self.output_dir / 'svd_model.pkl', 'wb') as f:
                pickle.dump(svd_model, f)

        # Save configuration
        config = {
            'n_movies': len(df),
            'use_svd': self.use_svd,
            'n_components': self.n_components if svd_model is not None else None,
            'format': 'neighbors' if neighbors is not None else 'similarity_matrix',
            'top_k': self.top_k if neighbors is not None else None,
            'matrix_shape': matrix_shape,
            'dataset': 'TMDB 2023',
        }
        with open(self.output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Model saved to {self.output_dir}")

        # Print summary
        total_size = sum(
            f.stat().st_size
            for f in self.output_dir.iterdir()
            if f.is_file()
        ) / 1024**2
        print(f"Total model size: {total_size:.1f} MB")

    def train(self, data_path, quality_threshold='medium', max_movies=None,
              legacy_matrix=False):
        """
        Complete training pipeline

        Args:
            data_path: Path to CSV file or directory containing it
            quality_threshold: 'low', 'medium', or 'high'
            max_movies: Limit number of movies (None = all)
            legacy_matrix: Save the full N x N similarity matrix instead of
                top-K neighbours. Only useful for tooling that expects the
                older format; needs vastly more memory and disk.

        Returns:
            (df, neighbors_or_matrix)
        """
        print("=" * 80)
        print("🎬 TMDB Movie Recommendation System Training")
        print("=" * 80)

        df = self.load_data(data_path)
        df = self.clean_and_engineer_features(df, quality_threshold)

        if max_movies and len(df) > max_movies:
            df = df.head(max_movies).copy()
            print(f"Limited to top {max_movies} movies by quality score")
            df = df.reset_index(drop=True)

        tfidf_matrix, tfidf_vectorizer = self.build_tfidf_matrix(df)

        if legacy_matrix:
            similarity_matrix, svd_model = self.compute_similarity_matrix(tfidf_matrix)
            self.save_model(df, tfidf_vectorizer, svd_model,
                            similarity_matrix=similarity_matrix)
            result = similarity_matrix
        else:
            features, svd_model = self.build_features(tfidf_matrix)
            neighbors = self.compute_neighbors(features)
            self.save_model(df, tfidf_vectorizer, svd_model, neighbors=neighbors)
            result = neighbors

        print("=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)

        return df, result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the movie recommendation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'data_path',
        help="Path to the dataset CSV (or a directory containing "
             "TMDB_movie_dataset_v11.csv)",
    )
    parser.add_argument('-o', '--output-dir', default='./models',
                        help="Where to write the model artifacts")
    parser.add_argument('-q', '--quality', default='medium',
                        choices=['low', 'medium', 'high'],
                        help="Minimum vote count: low=5, medium=50, high=500")
    parser.add_argument('-m', '--max-movies', type=int, default=50000,
                        help="Cap the catalogue to the top N movies by quality "
                             "score (0 means no cap)")
    parser.add_argument('-k', '--top-k', type=int, default=50,
                        help="Neighbours stored per movie")
    parser.add_argument('--components', type=int, default=500,
                        help="SVD latent dimensions")
    parser.add_argument('--no-svd', action='store_true',
                        help="Skip SVD and use the raw TF-IDF vectors")
    parser.add_argument('--chunk-size', type=int, default=1024,
                        help="Rows compared at a time; lower it if memory is tight")
    parser.add_argument('--legacy-matrix', action='store_true',
                        help="Save the full N x N similarity matrix (memory heavy)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    trainer = MovieRecommenderTrainer(
        output_dir=args.output_dir,
        use_dimensionality_reduction=not args.no_svd,
        n_components=args.components,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )

    df, _ = trainer.train(
        args.data_path,
        quality_threshold=args.quality,
        max_movies=args.max_movies or None,
        legacy_matrix=args.legacy_matrix,
    )

    print("\n📊 Final statistics:")
    print(f"   Movies in model: {len(df):,}")
    print(f"   Output directory: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
