import numpy as np
import pandas as pd
import json
import torch
import networkx as nx

import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
AN_TO_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def load_smiles():
    df = pd.read_csv(f'data/zinc250k.csv')
    with open(f'data/valid_idx_zinc250k.json') as f:
        test_idx = json.load(f)
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    return list(df['smiles'].loc[train_idx]), list(df['smiles'].loc[test_idx])


def get_novelty_in_df(df):
    train_smiles, _ = load_smiles()
    train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    
    if 'sim' not in df.keys():
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['mol']]
        train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
        max_sims = []
        for i in range(len(gen_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], train_fps)
            max_sims.append(max(sims))
        df['sim'] = max_sims


def gen_mol(x, adj, dataset, largest_connected_comp=True):    
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()

    if dataset == 'QM9':
        atomic_num_list = [6, 7, 8, 9, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct: num_no_correct += 1
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        mols.append(vcmol)
    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def construct_mol(x, adj, atomic_num_list): # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]              # 9,
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)            # 9, 9
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1                                # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol
