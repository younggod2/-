import os
import re
import json
import pickle
import pandas as pd
from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_1to3
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.subgraphs import extract_subgraph_from_point
from graphein.molecule.edges.distance import compute_distmat

import logging
csm_logger = logging.getLogger('gen_csm')
csm_logger.setLevel(logging.INFO)
handler2 = logging.FileHandler('./feature_engineering/logs/gen_csm_AT.log', mode='w')
formatter2 = logging.Formatter("%(levelname)s %(message)s")
handler2.setFormatter(formatter2)
csm_logger.addHandler(handler2)


# Загружаем словарь с фармакофорами для атомов в АК
with open('./feature_engineering/CSM/AA_atom_pharm_dict.json', 'r') as file:
    AA_atom_pharm_dict = json.load(file) 


def extractResidueEnvironment(pdb_id, mutation, chain, path2pdbs, res_env_radius, pdb_graph_dict) -> pd.DataFrame:
    """
    Создает подграф вокруг мутируемого остатка, сохраняет df в pkl файл

    - pdb_id (str): Идентификатор PDB структуры белка.
    - aa (str): Изначальная АК
    - num_aa (str): Номер мутируемой АК
    - mut_aa (str): АК, на которую происходит замена
    - chain (str): Цепь, к которой относится мутирующая АК.

    Пример использования:
    >>> residue_environment = extractResidueEnvironment('1AHW', 'A178D', 'D', '178', 'A', 'C')
    """

    # Если окружение было посчитано ранее, то достаем его из файла
    env_path = f'./feature_engineering/CSM/res_env/{pdb_id}_{mutation}_env.pkl'
    if os.path.isfile(env_path):
        with open(env_path, 'rb') as file:
            residue_environment = pickle.load(file)
        csm_logger.info(f'Residue environment for {pdb_id},{mutation},{chain} было взято из файла {env_path}')
        return residue_environment

    # Функция для разделения строки mutation
    def split_mutation_AAA(mutation):
        pattern = re.compile(r"([A-Za-z]+)(\d+)([A-Za-z]+)")
        match = pattern.match(mutation)

        aa = match.groups()[0].upper()
        num_aa = match.groups()[1]
        mut_aa = match.groups()[2].upper()
        if len(aa) == 1:
            aa = protein_letters_1to3[aa].upper() # F –> Phe –> PHE
        if len(mut_aa) == 2 or len(mut_aa) == 4: # если попался номер с буквой
            num_aa += mut_aa[0]
            mut_aa = mut_aa[1:]
        if len(mut_aa) == 1:
            mut_aa = protein_letters_1to3[mut_aa].upper() # A –> Ala –> ALA

        return aa, num_aa, mut_aa


    aa, num_aa, mut_aa = split_mutation_AAA(mutation)


    if num_aa[-1].isalpha(): # 100B —> 100 + B
        digit_aa, letter_aa = num_aa[:-1], num_aa[-1]
    else:
        digit_aa, letter_aa = num_aa, 'B' # здесь неинтуитивно: 'B' потому что графеин берет ротамер B по умолчанию, и пишет в ту же колонку, куда и букву от номера аминокислоты

    dataset = re.search(r'/([^\/]+)$', path2pdbs).group(1) # '../datasets/SKEMPI2_PDBs' —> 'SKEMPI2_PDBs'

    # Задаем параметры для создания графа: узлы – атомы, ребра – ковалентные связи
    params_to_change = {"granularity": "atom", "edge_construction_functions": [], 
                        "insertions": True}
    config = ProteinGraphConfig(**params_to_change)


    # Создаем граф для белка, если не создавали ранее
    if f'{pdb_id}_from_{dataset}' not in pdb_graph_dict.keys():
        try:
            graph = construct_graph(config=config, path=f'{path2pdbs}/{pdb_id}.pdb')
            pdb_graph_dict[f'{pdb_id}_from_{dataset}'] = graph
            csm_logger.info(f'Graph для {pdb_id} был построен из {path2pdbs}!')
        except:
            csm_logger.exception(f'Не удалось построить graph для {pdb_id} из {path2pdbs}')
    else:
        graph = pdb_graph_dict[f'{pdb_id}_from_{dataset}']
    

    # Поиск координат С-альфа атома мутируемой АК, он будет центром подграфа
    mut_center_was_found = False
    for node, data in graph.nodes(data=True):
        if f'{chain}:{aa}:{num_aa}:CA' in node or f'{chain}:{mut_aa}:{num_aa}:CA' in node  or f'{chain}:{aa}:{digit_aa}:{letter_aa}:CA' in node or f'{chain}:{mut_aa}:{digit_aa}:{letter_aa}:CA' in node:
            mut_center = data['coords']
            csm_logger.info(f'mut_center для {pdb_id},{mutation},{chain} был найден!')
            mut_center_was_found = True
            break
    if not mut_center_was_found:
        csm_logger.exception(f'Не удалось найти mut_center для {pdb_id},{mutation},{chain}!')
    

    # Выделяем подграф residue_environment вокруг мутируемой АК и сохраняем в виде DataFrame
    residue_environment = extract_subgraph_from_point(graph, centre_point=(mut_center), radius=res_env_radius)
    residue_environment = residue_environment.graph["pdb_df"]
    csm_logger.info(f'subgraph для {pdb_id},{mutation},{chain} был выделен!')

    # Удаляем атомы мутируемой АК из подграфа –> не нужно считать CSM для обратной структуры
    mask = residue_environment['residue_id'].isin([f'{chain}:{aa}:{num_aa}', f'{chain}:{mut_aa}:{num_aa}'])
    residue_environment = residue_environment[~mask]

    # Добавляем в датафрейм колонки с фармакофорами, используя AA_atom_pharm_dict
    residue_environment['Pharmacophore'] = residue_environment.apply(
        lambda x: AA_atom_pharm_dict[x['residue_name']][x['atom_name']], axis=1)

    # Сохранение датафрейма в файл pkl
    os.makedirs('./CSM/res_env', exist_ok=True)
    residue_environment.to_pickle(env_path)
    csm_logger.info(f'residue_environment для {pdb_id},{mutation},{chain} было сохранено в {env_path}!')

    return residue_environment


def calculateAtomicPairwiseDist(res_env) -> pd.DataFrame:

    coords = res_env.filter(like='_coord')
    distMatrix = compute_distmat(coords)

    pharmacophores = list(res_env['Pharmacophore'])
                          
    # Преобразование списка pharmacophores в строковый тип данных, это необходимо чтобы в getFrequency искать класс фармакофора в индексах (одному атому соответствуют несколько)
    pharmacophores_str = pd.Series(pharmacophores).astype(str)

    # Создание DataFrame с индексами, означающими фармакофор, в строковом типе данных
    distMatrix = pd.DataFrame(distMatrix, index=pharmacophores_str, 
                                        columns=pharmacophores_str)

    return distMatrix


def getFrequency(distMatrix, dist, classes, Dstep):

    cls1, cls2 = classes

    distMatrix.index = distMatrix.index.astype(str)
    distMatrix.columns = distMatrix.columns.astype(str)

    # Проверка наличия фармакофоров в окружении АК, если какого-то класса нет — возвращаем ноль
    if not any(distMatrix.index.str.contains(cls1)) or not any(distMatrix.columns.str.contains(cls2)):
        return 0
        
    # Выбираем только значения на пересечении классов
    # Используем метод str.contains для фильтрации строк и столбцов
    filt_distMatrix = distMatrix.loc[distMatrix.index.str.contains(cls1), distMatrix.columns.str.contains(cls2)]
    
    # дополнительно проверяем > 0.001
    frequency_matrix = ((filt_distMatrix > 0.001) & (filt_distMatrix > (dist - Dstep)) & (filt_distMatrix < dist))
    
    # sum().sum() — первый раз по строкам, второй раз суммы строк
    frequency = int(frequency_matrix.sum().sum())
    
    # если считаем для атомов одного класса, частоту делим пополам, т.к. dist(A1,A2) == dist(A2,А1)
    if cls1 == cls2:
        frequency /= 2

    return frequency


def Generate_CSM(pdb_id: str, mutation: str, chain: str, path2pdbs: str, res_env_radius, Dmin, Dmax, Dstep, pdb_graph_dict: dict) -> list:
    """
    Пример использования:
    >>> Generate_CSM('1AHW', 'D178A', 'C', '../datasets/SKEMPI2_PDBs_from_rcsb', 10, 4, 21, 4, {})
    """

    residue_environment = extractResidueEnvironment(pdb_id, mutation, chain, path2pdbs, res_env_radius, pdb_graph_dict)
    distMatrix = calculateAtomicPairwiseDist(residue_environment)
    csm_logger.info(f'distMatrix для {pdb_id},{mutation},{chain} была посчитана!')
    
    CSM_row = []

    for dist in range(Dmin, Dmax, Dstep):
        for classes in product(('Hyd','Pos','Neg','Acc','Don','Aro','Sul','Neu'), repeat=2): # берем комбинацию из 2 классов с повторениями
            frequency = getFrequency(distMatrix, dist, classes, Dstep)
            CSM_row.append(frequency)

    return CSM_row, pdb_graph_dict
