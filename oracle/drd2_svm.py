import os
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

_MODEL_PATH = os.environ.get(
    "DRD2_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "drd2", "clf_py36.pkl"),
)

_clf = None


class _CompatUnpickler(pickle.Unpickler):
    MODULE_MAP = {
        "sklearn.svm.classes": "sklearn.svm._classes",
        "sklearn.svm.base": "sklearn.svm._base",
        "sklearn.linear_model.logistic": "sklearn.linear_model._logistic",
        "sklearn.ensemble.forest": "sklearn.ensemble._forest",
        "sklearn.ensemble.gradient_boosting": "sklearn.ensemble._gb",
        "sklearn.tree.tree": "sklearn.tree._tree",
        "sklearn.neighbors.typedefs": "sklearn.neighbors._typedefs",
        "sklearn.neighbors.quad_tree": "sklearn.neighbors._quad_tree",
        "sklearn.utils.linear_assignment_": "sklearn.utils._linear_assignment",
    }

    def find_class(self, module, name):
        module = self.MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def _patch_old_sklearn(clf):
    if not hasattr(clf, "_support_vectors_"):
        d = clf.__dict__
        rename = {
            "dual_coef_": "_dual_coef_",
            "support_vectors_": "_support_vectors_",
            "support_": "_support",
            "n_support_": "_n_support",
            "intercept_": "_intercept_",
            "probA_": "_probA",
            "probB_": "_probB",
        }
        for old, new in rename.items():
            if old in d and new not in d:
                setattr(clf, new, d[old])
    return clf


def _load_model(path=_MODEL_PATH):
    global _clf
    if _clf is not None:
        return _clf
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DRD2 oracle model not found: {path}\n"
            "Please download the DRD2 SVM weights (clf_py36.pkl) and place them "
            "under models/drd2/ (see README.md)."
        )
    with open(path, "rb") as f:
        _clf = _CompatUnpickler(f).load()
    _patch_old_sklearn(_clf)
    return _clf


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    nfp = np.zeros((1, 2048), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % 2048
        nfp[0, nidx] += int(v)
    return nfp


def predict_drd2(smiles_list, model_path=_MODEL_PATH):
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    if len(smiles_list) == 0:
        return []
    clf = _load_model(model_path)

    fps = []
    mask = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mask.append(int(mol is not None))
        fps.append(fingerprints_from_mol(mol) if mol else np.zeros((1, 2048)))

    fps = np.concatenate(fps, axis=0)
    scores = clf.predict_proba(fps)[:, 1]
    scores = scores * np.array(mask)
    return scores.astype(np.float32).tolist()
