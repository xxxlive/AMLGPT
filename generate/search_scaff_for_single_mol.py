import numpy as np
import pandas as pd
from collections import defaultdict

import scaffoldgraph as sg
import networkx as nx
import os
from rdkit.Chem import Draw
from rdkit import Chem
from utils import get_mol, generate_image
from tqdm import tqdm

# tutorial of scaffoldgraph:
# https://github.com/UCLCheminformatics/ScaffoldGraph/tree/main
def filter_sca(sca):
    mol = Chem.MolFromSmiles(sca)
    if mol == None:
        return None
        print("WUXIAO")
    ri = mol.GetRingInfo()
    benzene_smarts = Chem.MolFromSmiles("c1ccccc1")
    if ri.NumRings() == 1 and mol.HasSubstructMatch(benzene_smarts) == True:
        return None
    elif mol.GetNumHeavyAtoms() > 20:
        return None
    elif Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) > 3:
        return None
    else:
        return True


def mol_if_equal_sca(mol,sca):
    S_sca = []
    #Chem.Kekulize(mol)
    smile = get_mol(mol)
    sca_mol = get_mol(sca)
    # smile =  Chem.MolFromSmiles(mol)
    # sca_mol = Chem.MolFromSmarts(sca)
    if smile == None or sca_mol == None :
        return False
    else:
        n_atoms = smile.GetNumAtoms()
        index = smile.GetSubstructMatch(sca_mol)
        for i in range(n_atoms):
            if i in index:
                S_sca.append(1)
            else:
                S_sca.append(0)
        arr = np.array(S_sca)
        if (arr == 1).all() == True or (arr == 0).all() == True:
            return False
        else:
            return True


def query_scaffold(smi_file, smi):
    network = sg.ScaffoldNetwork.from_smiles_file(smi_file)
    n_scaffolds = network.num_scaffold_nodes
    n_molecules = network.num_molecule_nodes

    scaffolds = list(network.get_scaffold_nodes())
    molecules = list(network.get_molecule_nodes())
    counts = network.get_hierarchy_sizes()  # returns a collections Counter object
    lists = sorted(counts.items())

    sca = []
    for pubchem_id in molecules:
        predecessors = list(nx.bfs_tree(network, pubchem_id, reverse=True))
        smi = network.nodes[predecessors[0]]['smiles']
        # 获取smiles_list中元素为smile的索引号
        max_len_scaff = 0

        for i in range(1, len(predecessors)):
            if filter_sca(predecessors[i]) is not None:
                sca.append(predecessors[i])

        if len(sca) != 0:
            for tmp_s in tqdm(sca):
                # random_sca = random.choice(sca)
                if mol_if_equal_sca(smi, tmp_s) is True:
                    sca.append(tmp_s)
    # filtered_mol_scaff_dict = defaultdict(list)

    return sca


if __name__ == "__main__":
    smi = 'COc1ccc(-c2c(C#N)[n+]([O-])c3ccccc3[n+]2[O-])cc1'
    smi_file = 'single_mol.smi'
    with open(smi_file, 'w') as f:
        f.write(smi)
    scaffs = query_scaffold(smi_file, smi)
    print(scaffs)