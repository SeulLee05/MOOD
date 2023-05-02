from rdkit import Chem
from scorer.scorer import get_scores


smiles = ['O=C6C=C(C4=CC1(CC(=O)CC2CCCC12)C3=C(C=CC=C3)C4=O)C5=C(C=C(C=C5)F)C6',
          'O=C(NC1=C(Cc2ccccc2)C=CC=CC=CC=C(F)C(F)=C1)c1ccccc1',
          'Cc1ccc(-c2cccccccc(C(F)=CC=C3CCO3)c2)c2c1=CC=2',
          'Cc1cccc(Cc2ccccc3ccc(c2)CC3)c1',
          'CCN1CC(CC(=O)C=Cc2ccccccccc2)Cc2cccc(F)c21']
print(get_scores('sa', [Chem.MolFromSmiles(s) for s in smiles]))