# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py

from rdkit import Chem
import rdkit.Chem.QED as QED
import numpy as np

from scorer import sa_scorer
from scorer.docking import get_dockingvina

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def standardize_mols(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        print('standardize_mols error')
        return None


def standardize_smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception:
        print('standardize_smiles error')
        return None


def get_docking_scores(target, mols, verbose=False):
    dockingvina = get_dockingvina(target)

    smiles = [standardize_smiles(mol) for mol in mols]
    smiles_valid = [smi for smi in smiles if smi is not None]
    
    scores = - np.array(dockingvina.predict(smiles_valid))
    if verbose:
        print(f'Number of docking errors: {sum(scores < -99)} / {len(scores)}')
    scores = list(np.clip(scores, 0, None))

    if None in smiles:
        scores = [scores.pop(0) if smi is not None else 0. for smi in smiles]

    return scores


def get_scores(objective, mols, standardize=True):
    if objective in ['jak2', 'braf', 'fa7', 'parp1', '5ht1b']:
        scores = get_docking_scores(objective, mols, True)
    else:
        if standardize:
            mols = [standardize_mols(mol) for mol in mols]
        mols_valid = [mol for mol in mols if mol is not None]
        
        scores = [get_score(objective, mol) for mol in mols_valid]
        scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
    
    return scores


def get_score(objective, mol):
    try:
        if objective == 'qed':
            return QED.qed(mol)
        elif objective == 'sa':
            x = sa_scorer.calculateScore(mol)
            return (10. - x) / 9.   # normalized to [0, 1]
        else:
            raise NotImplementedError
    except (ValueError, ZeroDivisionError):
        return 0.
