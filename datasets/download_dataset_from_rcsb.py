import os
import pandas as pd
from Bio.PDB import PDBList

destination_folder = './SKEMPI2_PDBs_from_rcsb'
source_table = './mCSM-AB2_dataset.csv'

def download_pdb(pdb_id):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb_id, pdir=destination_folder, file_format='pdb')
    ent_filename = f'{destination_folder}/pdb{pdb_id}.ent'
    pdb_filename = f'{destination_folder}/{pdb_id}.pdb'
    os.rename(ent_filename, pdb_filename)

for pdb_id in pd.read_csv(source_table)['PDB'].unique():
    download_pdb(pdb_id)
