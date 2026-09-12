"""
Tests for the training and inference pipeline.

Deliberately narrow. These cover the parsing and neighbour-selection logic --
the places where mistakes are silent rather than loud, and where this project
has actually shipped bugs: genres vanishing because a numpy array is not a
list, and release years read off the wrong end of an ISO date.

scikit-learn and nltk are only needed for training, not for serving, so they
are not in requirements.txt. These tests skip themselves when those are absent.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:  # training extras -- see requirements-train.txt
    from training.train import MovieRecommenderTrainer
    TRAINER_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    TRAINER_AVAILABLE = False

from training.infer import as_list, release_year

requires_trainer = unittest.skipUnless(
    TRAINER_AVAILABLE, "scikit-learn/nltk not installed (pip install -r requirements-train.txt)"
)


class MetadataHelperTests(unittest.TestCase):
    """Helpers that turn messy dataset cells into something usable."""

    def test_as_list_handles_numpy_arrays(self):
        """Parquet list columns load as numpy arrays, not Python lists.

        An `isinstance(value, list)` check here silently drops every genre,
        which is exactly the bug this guards against.
        """
        self.assertEqual(as_list(np.array(['Action', 'Drama'])), ['Action', 'Drama'])
        self.assertEqual(as_list(['Action']), ['Action'])
        self.assertEqual(as_list(('Action',)), ['Action'])

    def test_as_list_handles_missing_values(self):
        self.assertEqual(as_list(None), [])
        self.assertEqual(as_list(float('nan')), [])
        self.assertEqual(as_list('Action'), [])  # a bare string is not a list of genres

    def test_release_year_reads_the_year_not_the_day(self):
        """`1999-03-30`.split('-')[-1] is 30, which is not a year."""
        self.assertEqual(release_year('1999-03-30'), 1999)
        self.assertEqual(release_year('2014-11-05'), 2014)
        self.assertEqual(release_year('1975'), 1975)

    def test_release_year_handles_missing_values(self):
        self.assertIsNone(release_year(''))
        self.assertIsNone(release_year(None))
        self.assertIsNone(release_year('n/a'))


@requires_trainer
class ParseJsonColumnTests(unittest.TestCase):
    """The dataset encodes genres/keywords inconsistently between exports."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='trainer-'))
        cls.trainer = MovieRecommenderTrainer(output_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_list_of_dicts(self):
        raw = '[{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}]'
        self.assertEqual(self.trainer.parse_json_column(raw), ['Action', 'Drama'])

    def test_comma_separated_fallback(self):
        """TMDB's v11 CSV uses plain comma-separated names, not JSON."""
        self.assertEqual(self.trainer.parse_json_column('Action, Drama'), ['Action', 'Drama'])

    def test_already_a_list(self):
        self.assertEqual(self.trainer.parse_json_column([{'name': 'Action'}]), ['Action'])

    def test_empty_and_missing(self):
        for value in ('', '[]', None, float('nan')):
            self.assertEqual(self.trainer.parse_json_column(value), [], repr(value))

    def test_malformed_input_does_not_raise(self):
        self.assertIsInstance(self.trainer.parse_json_column('{not valid at all'), list)


@requires_trainer
class ComputeNeighborsTests(unittest.TestCase):
    """Top-K neighbour selection -- the core of what training produces."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='trainer-'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _trainer(self, top_k=3, chunk_size=2):
        return MovieRecommenderTrainer(
            output_dir=self.tmp, use_dimensionality_reduction=False,
            top_k=top_k, chunk_size=chunk_size,
        )

    @staticmethod
    def _features():
        """Five points on a line: nearer indices are more similar."""
        return np.array([[1.0, 0.0], [0.9, 0.1], [0.7, 0.3],
                         [0.3, 0.7], [0.0, 1.0]], dtype=np.float32)

    def test_shape_and_dtypes(self):
        idx, scores = self._trainer().compute_neighbors(self._features())
        self.assertEqual(idx.shape, (5, 3))
        self.assertEqual(scores.shape, (5, 3))
        self.assertEqual(idx.dtype, np.int32)

    def test_never_recommends_the_movie_itself(self):
        idx, _ = self._trainer().compute_neighbors(self._features())
        for row, neighbours in enumerate(idx):
            self.assertNotIn(row, neighbours.tolist(), f"row {row} is its own neighbour")

    def test_scores_are_sorted_best_first(self):
        _, scores = self._trainer().compute_neighbors(self._features())
        for row in scores:
            self.assertEqual(row.tolist(), sorted(row.tolist(), reverse=True))

    def test_nearest_neighbour_is_the_adjacent_point(self):
        idx, _ = self._trainer().compute_neighbors(self._features())
        self.assertEqual(idx[0][0], 1)   # closest to [1.0, 0.0]
        self.assertEqual(idx[4][0], 3)   # closest to [0.0, 1.0]

    def test_chunking_does_not_change_the_result(self):
        """Chunk size is a memory knob; it must not affect output."""
        features = self._features()
        small = self._trainer(chunk_size=2).compute_neighbors(features)
        big = self._trainer(chunk_size=100).compute_neighbors(features)
        np.testing.assert_array_equal(small[0], big[0])
        np.testing.assert_allclose(small[1], big[1], rtol=1e-6)

    def test_top_k_is_capped_at_the_catalogue_size(self):
        """Asking for more neighbours than there are movies must not blow up."""
        idx, _ = self._trainer(top_k=50).compute_neighbors(self._features())
        self.assertEqual(idx.shape[1], 4)  # 5 movies - itself

    def test_single_movie_is_rejected_clearly(self):
        with self.assertRaises(ValueError):
            self._trainer().compute_neighbors(np.array([[1.0, 0.0]], dtype=np.float32))


class InferenceLoadingTests(unittest.TestCase):
    """The inference engine must read every stored format identically."""

    @classmethod
    def setUpClass(cls):
        from recommender.tests import build_model  # shared synthetic fixture
        cls.build_model = staticmethod(build_model)
        cls.neighbors_dir = Path(tempfile.mkdtemp(prefix='infer-nb-'))
        cls.matrix_dir = Path(tempfile.mkdtemp(prefix='infer-mx-'))
        build_model(cls.neighbors_dir, with_neighbors=True)
        build_model(cls.matrix_dir, with_neighbors=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.neighbors_dir, ignore_errors=True)
        shutil.rmtree(cls.matrix_dir, ignore_errors=True)

    def _recommender(self, directory):
        from training.infer import MovieRecommender
        return MovieRecommender(model_dir=directory)

    def test_both_storage_formats_agree(self):
        titles = []
        for directory in (self.neighbors_dir, self.matrix_dir):
            result = self._recommender(directory).get_recommendations(
                'The Matrix', n_recommendations=3)
            titles.append([m['title'] for m in result['recommendations']])
        self.assertEqual(titles[0], titles[1])

    def test_dense_matrix_is_not_read_into_memory_whole(self):
        """A full similarity matrix is ~10 GB at 50k movies; it must be mapped."""
        recommender = self._recommender(self.matrix_dir)
        self.assertIsInstance(recommender.similarity_matrix, np.memmap)

    def test_genres_survive_the_round_trip(self):
        result = self._recommender(self.neighbors_dir).get_recommendations(
            'The Matrix', n_recommendations=3)
        for movie in result['recommendations']:
            self.assertTrue(movie['genres'], f"{movie['title']} lost its genres")

    def test_missing_model_raises_a_clear_error(self):
        empty = Path(tempfile.mkdtemp(prefix='infer-empty-'))
        try:
            with self.assertRaises(FileNotFoundError):
                self._recommender(empty)
        finally:
            shutil.rmtree(empty, ignore_errors=True)
