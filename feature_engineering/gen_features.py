import re
import json
import pandas as pd
import numpy as np
from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
from tqdm.notebook import tqdm
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import logging
feat_logger = logging.getLogger('gen_features')
feat_logger.setLevel(logging.INFO)
handler2 = logging.FileHandler('./feature_engineering/logs/gen_features.log', mode='w')
formatter2 = logging.Formatter("%(levelname)s %(message)s")
handler2.setFormatter(formatter2)
feat_logger.addHandler(handler2)

from feature_engineering.res_biochem import residueFeature
from feature_engineering.CSM.gen_csm_AT import Generate_CSM


# Функция для разделения строки mutation
def split_mutation_A(mutation):
    pattern = re.compile(r"([A-Za-z]+)(\d+)([A-Za-z]+)")
    match = pattern.match(mutation)

    aa = match.groups()[0].capitalize()
    num_aa = match.groups()[1]
    mut_aa = match.groups()[2].capitalize()
    if len(aa) == 3:
        aa = protein_letters_3to1[aa].upper() # PHE –> F
    if len(mut_aa) == 2 or len(mut_aa) == 4: # если попался номер с буквой
        num_aa += mut_aa[0]
        mut_aa = mut_aa[1:].capitalize()
    if len(mut_aa) == 3:
        mut_aa = protein_letters_3to1[mut_aa].upper() # ALA –> A

    return aa, num_aa, mut_aa


# Загружаем словарь с вектором фармакофоров для каждой АК
with open('./feature_engineering/CSM/AA_vector_pharm_dict.json', 'r') as file:
    AA_vector_pharm_dict = json.load(file)

# Переводим все списки в значениях словаря в np.array для удобства вычитания друг из друга в дальнейшем
for key in AA_vector_pharm_dict:
    AA_vector_pharm_dict[key] = np.array(AA_vector_pharm_dict[key])

#!!! Clark et al.


def GenerateFeatures(path2mutations, paths2pdbs, res_env_radius=10, Dmin=4, Dmax=21, Dstep=4):
    Features = []
    MutationSet = pd.read_csv(path2mutations)
    pdb_graph_dict = {}
    paths2pdbs = [paths2pdbs] # изменю, чтобы на вход был не список, а просто путь

    with tqdm(total=len(MutationSet), desc="Mutations") as progressbar:
        for i, (index, Mutation) in enumerate(MutationSet.iterrows()):

            pdb_id = Mutation['PDB']
            mutation = Mutation['mutation']
            chain = Mutation['chain']
            aa, num_aa, mut_aa = split_mutation_A(mutation) # ... –> F+149+A
            feat_logger.info(f'Выделяем фичи для {pdb_id},{mutation},{chain}... 🏃')


            for path in paths2pdbs:
            ##################
            ..................
            ##################
  
            pharmacophore_count = AA_vector_pharm_dict[mut_aa] - AA_vector_pharm_dict[aa]
            residue_feature = residueFeature(mut_aa) - residueFeature(aa)
                
            FeaturesRow = [
                *CSM_row, 
                *pharmacophore_count,
                *residue_feature
                ]
            Features.append(FeaturesRow)
            feat_logger.info(f'Успех! 😀')

            ##################
            ..................
            ##################

            progressbar.update()


    combinations = list(product(('Hyd','Pos','Neg','Acc','Don','Aro','Sul','Neu'), repeat=2))
    column_names = [f"{cls1}_{cls2}_{dist}" for dist in range(Dmin, Dmax, Dstep) for cls1, cls2 in combinations]
    column_names.extend(['∆Hyd', '∆Pos', '∆Neg', '∆Acc', '∆Don', '∆Aro', '∆Sul', '∆Neu'])
    column_names.extend(['∆AAvolume', '∆AAhydropathy', '∆AAarea', '∆AAweight', '∆AAcharge', '∆AAflexibily', '∆AAchemical', '∆AAsize', '∆AAhbonds'])
    Features = pd.DataFrame(Features, columns=column_names)

    return Features
