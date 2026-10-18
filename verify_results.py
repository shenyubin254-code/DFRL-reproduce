"""Recompute saved metrics without loading either LLM or generating molecules."""
import argparse
import csv
import json
import math
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from oracle.drd2_svm import predict_drd2


def verify(path, data_path, threshold, strict=False):
    RDLogger.DisableLog('rdApp.*')
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    with Path(data_path).open(newline='', encoding='utf-8') as stream:
        inputs = [row['sd_smiles_Q'] for row in csv.DictReader(stream)]
    if not rows or [r['input_smiles'] for r in rows] != inputs:
        raise ValueError('Result rows/order do not match the full test set')
    errors = []
    for i, row in enumerate(rows):
        mols = [Chem.MolFromSmiles(row[k]) for k in ('input_smiles', 'optimized_smiles')]
        if any(m is None for m in mols):
            errors.append([i, 'invalid SMILES'])
            continue
        for prefix, mol, smiles in zip(('initial', 'optimized'), mols,
                                      (row['input_smiles'], row['optimized_smiles'])):
            expected = {'qed': QED.qed(mol), 'logp': Descriptors.MolLogP(mol),
                        'drd2': predict_drd2([smiles])[0]}
            for name, value in expected.items():
                saved = float(row[f'{prefix}_{name}'])
                if not math.isfinite(saved) or not math.isclose(saved, value, abs_tol=1e-7, rel_tol=1e-7):
                    errors.append([i, f'{prefix}_{name}', saved, value])
        similarity = DataStructs.TanimotoSimilarity(
            *[AllChem.GetMorganFingerprint(m, 2) for m in mols])
        if not math.isclose(float(row['similarity']), similarity, abs_tol=1e-12):
            errors.append([i, 'similarity'])
        if similarity < threshold or (strict and similarity <= threshold):
            errors.append([i, 'similarity constraint'])
    report = {'file': str(path), 'rows': len(rows), 'errors': errors,
              'similarity_rule': '>' if strict else '>=',
              'at_boundary': sum(float(r['similarity']) == threshold for r in rows),
              'qed_ge_0_9_percent': 100 * sum(float(r['optimized_qed']) >= .9 for r in rows) / len(rows),
              'qed_delta_ge_0_1_percent': 100 * sum(float(r['optimized_qed']) - float(r['initial_qed']) >= .1 for r in rows) / len(rows)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--data', default='data/standard_test_data/qed_test.csv')
    parser.add_argument('--similarity', type=float, required=True)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    report = verify(args.csv, args.data, args.similarity, args.strict)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report['errors']))
