from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd

from scorer.scorer import get_scores
from utils.mol_utils import get_novelty_in_df


def evaluate(protein, csv_dir, smiles, mols=None):
    df = pd.DataFrame()
    num_mols = len(smiles)

    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    validity = len(df) / num_mols

    if mols is None:
        df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        df['mol'] = mols

    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df)
    novelty = len(df[df['sim'] < 0.4]) / len(df)

    df = df.drop_duplicates(subset=['smiles'])

    df[protein] = get_scores(protein, df['mol'])
    df['qed'] = get_scores('qed', df['mol'])
    df['sa'] = get_scores('sa', df['mol'])

    del df['mol']
    df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    else: raise ValueError('Wrong target protein')

    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.4]
    df = df.sort_values(by=[protein], ascending=False)

    num_top5 = int(num_mols * 0.05)

    top_ds = df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std()
    hit = len(df[df[protein] > hit_thr]) / num_mols
    
    return {'validity': validity, 'uniqueness': uniqueness,
            'novelty': novelty, 'top_ds': top_ds, 'hit': hit}


def evaluate_baseline(df, csv_dir, protein):
    from moses.utils import get_mol
    
    num_mols = 3000

    drop_idx = []
    mols = []
    for i, smiles in enumerate(df['smiles']):
        mol = get_mol(smiles)
        if mol is None:
            drop_idx.append(i)
        else:
            mols.append(mol)
    df = df.drop(drop_idx)
    df['mol'] = mols
    print(f'Validity: {len(df) / num_mols}')
    
    df['smiles'] = [Chem.MolToSmiles(m) for m in df['mol']]      # canonicalize

    print(f'Uniqueness: {len(set(df["smiles"])) / len(df)}')
    get_novelty_in_df(df)
    print(f"Novelty (sim. < 0.4): {len(df[df['sim'] < 0.4]) / len(df)}")

    df = df.drop_duplicates(subset=['smiles'])

    if not protein in df.keys():
        df[protein] = get_scores(protein, df['mol'])

    if not 'qed' in df.keys():
        df['qed'] = get_scores('qed', df['mol'])

    if not 'sa' in df.keys():
        df['sa'] = get_scores('sa', df['mol'])

    del df['mol']
    df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf' : hit_thr = 10.3
    
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.4]
    df = df.sort_values(by=[protein], ascending=False)

    num_top5 = int(num_mols * 0.05)

    top_ds = df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std()
    hit = len(df[df[protein] > hit_thr]) / num_mols
    
    print(f'Novel top 5% DS (QED > 0.5, SA < 5, sim. < 0.4): '
          f'{top_ds[0]:.4f} ± {top_ds[1]:.4f}')
    print(f'Novel hit ratio (QED > 0.5, SA < 5, sim. < 0.4): {hit * 100:.4f} %')
