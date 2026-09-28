"""Verify eligibility polarity, exact top-k and rejection of undersized scenes."""
import argparse
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import prepare


class PreparationTest(unittest.TestCase):
    """Exercise preparation using a small deterministic dataset and an independent oracle."""

    def setUp(self):
        """Create isolated vectors and a bitset that excludes IDs divisible by three."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(17)
        self.train = prepare.normalize(rng.normal(size=(256, 12)))
        self.queries = prepare.normalize(rng.normal(size=(6, 12)))
        self.dataset = self.root / 'data.hdf5'
        with h5py.File(self.dataset, 'w') as data:
            data['train'] = self.train
            data['test'] = self.queries
        self.bitset = self.root / 'excluded.bits'
        excluded = np.arange(256) % 3 == 0
        np.packbits(excluded, bitorder='little').tofile(self.bitset)
        self.args = argparse.Namespace(topk=10, recall_nq=6, nq=2, concurrency=2,
                                       percentages=[5, 30], mask_mode='modulo', seed=17,
                                       bitset=[f'arbitrary={self.bitset}'])

    def test_exact_truth_and_mask_polarity(self):
        """Every scene must agree with an independent full cosine sort on eligible IDs."""
        directory = prepare.prepare(self.dataset, self.root / 'cache', self.args)
        train = prepare.mapped_matrix(directory / 'train.f32')
        queries = prepare.mapped_matrix(directory / 'test.f32')
        for name in (directory / 'scenes.txt').read_text().split():
            mask = np.fromfile(directory / f'{name}.mask', dtype='uint8')
            truth = np.fromfile(directory / f'{name}.gt', dtype='<i8').reshape(6, 10)
            valid = np.flatnonzero(mask)
            scores = queries.astype('float64') @ train[valid].astype('float64').T
            expected = valid[np.argsort(-scores, axis=1)[:, :10]]
            for actual, oracle in zip(truth, expected):
                self.assertEqual(set(actual), set(oracle))
        mask = np.fromfile(directory / 'custom_arbitrary.mask', dtype='uint8')
        np.testing.assert_array_equal(mask, np.arange(256) % 3 != 0)
        self.assertEqual(directory, prepare.prepare(self.dataset, self.root / 'cache', self.args))

    def test_too_few_eligible_vectors(self):
        """A scene with fewer than k eligible IDs must not publish a completion manifest."""
        self.args.percentages = [1]
        self.args.bitset = []
        cache = self.root / 'cache'
        with self.assertRaisesRegex(ValueError, 'fewer than topk'):
            prepare.prepare(self.dataset, cache, self.args)
        self.assertEqual(list(cache.glob('*/manifest.json')), [])

    def test_random_masks_are_reproducible(self):
        """Changing the selection mode yields deterministic masks with the configured cardinality."""
        self.args.mask_mode = 'random'
        masks = prepare.create_masks(256, self.args)
        repeated = prepare.create_masks(256, self.args)
        np.testing.assert_array_equal(masks['random_p30'], repeated['random_p30'])
        self.assertEqual(int(masks['random_p30'].sum()), int(256 * .30))


if __name__ == '__main__':
    unittest.main()
