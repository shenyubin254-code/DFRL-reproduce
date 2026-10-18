"""Small, local-only checkpoints for the optimization loop (not model training)."""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if os.name == 'posix':
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def experiment_identity(args):
    """Fail closed if inputs, code, weights, or runtime change on resume."""
    parameters = {k: v for k, v in vars(args).items()
                  if k not in ('resume', 'output_file')}
    root = Path(__file__).resolve().parent
    files = [root / name for name in
             ('optimize.py', 'utils.py', 'load_data.py', 'checkpointing.py',
              'oracle/drd2_svm.py')]
    files.append(Path(args.data_fname_test))
    files.append(Path(os.environ.get('DRD2_MODEL_PATH', root / 'models/drd2/clf_py36.pkl')))
    for directory in (args.spa_model_path, args.dcr_model_path):
        model_files = sorted(p for p in Path(directory).rglob('*') if p.is_file())
        if not model_files:
            raise ValueError(f'No local model files found: {directory}')
        files.extend(model_files)
    import transformers
    import rdkit
    import sklearn
    return {'parameters': parameters,
            'files': {str(p.resolve()): sha256(p) for p in files},
            'versions': [torch.__version__, transformers.__version__, rdkit.__version__,
                         np.__version__, sklearn.__version__],
            'cuda': torch.version.cuda,
            'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


STATE_NAMES = ('swarms swarms_qed swarms_logp swarms_drd2 init_fitness init_qed '
               'init_logp init_drd2 best_fitness best_qed best_logp best_drd2 '
               'best_similar best_smiles').split()


def save_checkpoint(path, namespace, identity, epoch, step, counters, journals):
    state = {'identity': identity, 'epoch': epoch, 'step': step, 'counters': counters,
             'arrays': {key: namespace[key] for key in STATE_NAMES},
             'random': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
             'offsets': {str(p): p.stat().st_size if p.exists() else 0 for p in journals}}
    atomic_bytes(path, pickle.dumps(state, protocol=4))


def load_checkpoint(path, identity):
    # Only load checkpoints produced locally by this experiment; pickle is trusted input.
    with Path(path).open('rb') as stream:
        state = pickle.load(stream)
    if state['identity'] != identity:
        raise ValueError('Resume refused: parameters, source, data, weights or runtime changed.')
    for name, offset in state['offsets'].items():
        path = Path(name)
        if (path.stat().st_size if path.exists() else 0) < offset:
            raise ValueError(f'Resume journal is shorter than checkpoint: {path}')
    return state


def restore_checkpoint(state, namespace):
    # Discard only uncommitted journal tails from an interrupted write, never older results.
    for name, offset in state['offsets'].items():
        path = Path(name)
        if path.exists():
            with path.open('r+b') as stream:
                stream.truncate(offset)
    namespace.update(state['arrays'])
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def mark_complete(path, output, identity, count):
    atomic_bytes(path, json.dumps({'identity': identity, 'num_molecules': count,
                                  'csv_sha256': sha256(output)}, indent=2).encode())
