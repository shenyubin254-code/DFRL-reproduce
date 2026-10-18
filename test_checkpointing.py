import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from checkpointing import STATE_NAMES, load_checkpoint, restore_checkpoint, save_checkpoint
from run_benchmark import aggregate_metrics


class ResumeTests(unittest.TestCase):
    def test_population_rng_and_journal_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            journal = path / 'history.csv'
            journal.write_bytes(b'committed\n')
            namespace = {name: [['CCO', 'CCC']] for name in STATE_NAMES}
            state_path = path / 'checkpoint.pkl'
            save_checkpoint(state_path, namespace, {'test': 1}, 2, 7, (3, 4, 5), [journal])
            expected = (random.random(), np.random.rand(), torch.rand(3))
            cuda_expected = torch.rand(3, device='cuda') if torch.cuda.is_available() else None
            journal.write_bytes(b'committed\nuncommitted partial row')
            state = load_checkpoint(state_path, {'test': 1})
            restored = {}
            restore_checkpoint(state, restored)
            self.assertEqual(restored, namespace)
            self.assertEqual((state['epoch'], state['step'], state['counters']), (2, 7, (3, 4, 5)))
            self.assertEqual(journal.read_bytes(), b'committed\n')
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.rand(), expected[1])
            self.assertTrue(torch.equal(torch.rand(3), expected[2]))
            if cuda_expected is not None:
                self.assertTrue(torch.equal(torch.rand(3, device='cuda'), cuda_expected))
            with self.assertRaises(ValueError):
                load_checkpoint(state_path, {'test': 2})
            journal.write_bytes(b'')
            with self.assertRaises(ValueError):
                load_checkpoint(state_path, {'test': 1})

    def test_single_seed_has_no_variance_estimate(self):
        rows = [{'objective': 'qed', 'min_similarity': .6, 'seed': 0,
                 'num_molecules': 800, 'mean_qed': .9}]
        result = aggregate_metrics(rows)[0]
        self.assertEqual(result['std'], '')
        self.assertEqual(result['runs'], 1)
        self.assertEqual(result['scope'], 'single_seed_no_variance_estimate')


if __name__ == '__main__':
    unittest.main()
