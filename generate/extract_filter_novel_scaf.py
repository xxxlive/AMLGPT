import numpy as np
import pandas as pd
from collections import defaultdict

import scaffoldgraph as sg
import networkx as nx
import os
from rdkit.Chem import Draw
from rdkit import Chem

import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm
import pickle as pkl
from pathlib import Path

import random
random.seed(42)

from Get_data_for_model import PreprocessData
from utils import get_mol, generate_image

# 获取当前路径
current_path = Path.cwd()
parent_path = current_path.parent

print("cur_path:", current_path)
print("par_path:", parent_path)


def filter_through_pubchem(mols_df, drop_missing_entries = True):
    
    Query_pub = PreprocessData(mols_df)
    
    try:
        dict_of_SMILES = pkl.load(open('dict_of_SMILES_and_CID', 'rb'))

    except OSError:
        print('Could not load dictionary.')
        commence_download = input('''Do you wish to retrieve CIDs from PubChem? [y/n]''')

        if commence_download in ['y', 'Yes', 'YES', 'yes']:
            unique_SMILES = Query_pub.dataframe.SMILES.unique()
            dict_of_SMILES = dict.fromkeys(unique_SMILES)

            for smile in tqdm(unique_SMILES):
                dict_of_SMILES[smile] = Query_pub.__GetCID(smile)

            Query_pub.dataframe['Pubchem_CID'] = Query_pub.dataframe.SMILES.apply(lambda x: dict_of_SMILES[x])
            pkl.dump(dict_of_SMILES, open('dict_of_SMILES_and_CID', 'wb'))

    Query_pub.dataframe = Query_pub.dataframe.reset_index().drop(columns='index', axis=1)
    CID_list = np.zeros(len(Query_pub.dataframe)).astype(int)
    dropable = []
    for idx, entry in tqdm(Query_pub.dataframe.SMILES.items()):
        if entry in dict_of_SMILES.keys():
            CID_list[idx] = dict_of_SMILES[entry]
            dropable.append(idx)
        else:
            pub_cid = Query_pub.get_cid(entry)
            if pub_cid is not None and pub_cid != 0:
                CID_list[idx] = pub_cid  # 0 corresponds to invalid CID
                dropable.append(idx)

    if drop_missing_entries == True:
        Query_pub.dataframe = Query_pub.dataframe.drop(dropable)
        CID_list = CID_list[CID_list == 0]
        print(f'Dropped {len(dropable)} entries from dataframe due to SMILES not having CID')
        Query_pub.dataframe['Pubchem_CID'] = CID_list.tolist()
    else:
        Query_pub.dataframe['Pubchem_CID'] = CID_list.tolist()

    Query_pub.dataframe = Query_pub.dataframe.reset_index().drop(columns='index', axis=1)

    return Query_pub.dataframe, Query_pub.dataframe.shape[0]


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



def load_mols_csv(gem_mol_csv):

    """
    1.1 load generated mols and transform the valid mols into .smi
            (construct a bijective mapping {mol--id})
    1.2 load the original dataset, extract the column of smiles
    """

    mols_df = pd.read_csv(gem_mol_csv)
    smiles_series = mols_df['smiles']

    # 构造 smiles -> 行号 的映射
    smi2ids = defaultdict()  # 使用 defaultdict 来存储每个 smiles 对应的所有行号

    smis_list = []
    # 通过 iterrows() 遍历 DataFrame，构建字典
    for idx, smi in smiles_series.items():
        smi2ids[smi] = idx  # idx 为行号，smile 为 smiles 列的值
        smis_list.append(smi + '\n')

    return smis_list, smi2ids, mols_df


def extract_scaffolds(mols_smi_file, original_mols_csv):
    """
    2.1 extract the scaffolds from the generated mols in .smi
            (construct dict {scaffold -- id})
    2.2 filter out the scaffolds not in the original dataset
    """
    original_mols_df = pd.read_csv(original_mols_csv)
    observed_scaffs = original_mols_df['SCAFFOLD_SMILES']

    network = sg.ScaffoldNetwork.from_smiles_file(mols_smi_file)
    n_scaffolds = network.num_scaffold_nodes
    n_molecules = network.num_molecule_nodes

    scaffolds = list(network.get_scaffold_nodes())
    molecules = list(network.get_molecule_nodes())
    counts = network.get_hierarchy_sizes()  # returns a collections Counter object
    lists = sorted(counts.items())

    num_new_scaffs = 0
    dict_mol_scaff = defaultdict(list)
    for pubchem_id in molecules:
        predecessors = list(nx.bfs_tree(network, pubchem_id, reverse=True))
        smi = network.nodes[predecessors[0]]['smiles']
        # 获取smiles_list中元素为smile的索引号
        max_len_scaff = 0

        sca = []
        for i in range(1, len(predecessors)):
            if filter_sca(predecessors[i]) is not None:
                sca.append(predecessors[i])

        if len(sca) != 0:
            for tmp_s in tqdm(sca):
                # random_sca = random.choice(sca)
                if tmp_s not in observed_scaffs.values:       # whether in observed sets
                    if mol_if_equal_sca(smi, tmp_s) is True:
                        # 如果已经有骨架了，取最长的scaffold
                        if smi not in dict_mol_scaff:
                            # dict_mol_scaff[smi].append(tmp_s)
                            max_len_scaff = len(tmp_s)
                        else:
                            tmp_len_scaff = len(tmp_s)
                            if tmp_len_scaff > max_len_scaff:
                                dict_mol_scaff[smi].pop()
                                max_len_scaff = tmp_len_scaff
                        dict_mol_scaff[smi].append(tmp_s)
                # f.write(smile + "," + random_sca + '\n')
                        num_new_scaffs += 1
    # filtered_mol_scaff_dict = defaultdict(list)

    return dict_mol_scaff, num_new_scaffs

"""
3.1 map the mols with scaffolds
"""
def collect_mols_scaffs_props(mols_df, dict_mol_scaff):
    smis_list = []
    scaffs_list = []
    logps_list = []
    qeds_list = []
    sas_list = []
    for smi in dict_mol_scaff.keys():
        row = mols_df[mols_df['smiles'] == smi]
        # 值不对,好像是 series, 应该用.iloc[0]来获取单个值
        logp = row['logp'].iloc[0]
        qed = row['qed'].iloc[0]
        sas = row['sas'].iloc[0]
        smis_list.append(smi)
        scaff = dict_mol_scaff[smi][0]
        scaffs_list.append(scaff)
        logps_list.append(logp)
        qeds_list.append(qed)
        sas_list.append(sas)
    collect_res_df = pd.DataFrame({'SMILES': smis_list,'scaffolds':scaffs_list,
                                   'logps':logps_list,'qeds':qeds_list, 'sas':sas_list})
    return collect_res_df

"""
4. random select examples and visualize them
# https://leonis.cc/sui-sui-nian/2023-01-03-rdkit-modify-search-molecule.html
# http://rdkit.chenzhaoqiang.com/mediaManual.html

"""

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20


def visualize_mols_scaffs_props(collect_csv, png_dir, n_samples=5):
    collect_mol_df = pd.read_csv(collect_csv)

    len_mols = collect_mol_df.shape[0]
    # random_samples = collect_mol_df.sample(n=n_samples, random_state=42)
    #
    # smis_mol_list = random_samples['SMILES'].tolist()
    # smis_scaff_list = random_samples['scaffolds'].tolist()

    smi_mol_list = []
    cnt = 0
    for i in range(len_mols):

        if cnt == n_samples:
            break

        record_demo = collect_mol_df.iloc[i]

        demo_smi = record_demo['SMILES']        # .iloc[0]
        demo_scff = record_demo['scaffolds']    # .iloc[0]

        print(demo_smi)
        print(demo_scff)

        # get mol
        demo_mol = Chem.MolFromSmiles(demo_smi)
        demo_frag = Chem.MolFromSmarts(demo_scff)
        # common atoms
        comm_atoms = demo_mol.GetSubstructMatches(demo_frag)

        # cannot find matched substructure
        if comm_atoms is None:
            continue
        else:
            smi_mol_list.append(demo_smi)
            cnt += 1

            comm_atoms = comm_atoms[0]
            # comm_bonds
            comm_bonds = set()

            # 获取与共同原子相连的边
            for atom_idx in comm_atoms:
                for neighbor in demo_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                    bond = demo_mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                    comm_bonds.add(bond.GetIdx())

            # Prepare colors
            atom_color = (0.95, 0.6, 0.0)
            bond_color = (0.95, 0.6, 0.0)
            radius = 0.3

            _ = generate_image(demo_mol, list(comm_atoms), list(comm_bonds), atom_color, bond_color, radius,
                           (400, 400), f'{png_dir}/demo_mol_{i}.png', False)

    with open(f'{png_dir}/demo_smis.txt', 'w') as f:
        f.writelines([smi+'\n' for smi in smi_mol_list])

    pass

"""
5. save them, done
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_name', default='guaca_gen.csv', help='dataset path')
    parser.add_argument("--original_mols_csv", default='guacamol2', help='original molecule dataset: MOSES / Guacamol')
    parser.add_argument('--filter_pubchem', action='store_true', default=False, help='filter out smiles in pubchem', )
    # set arguments
    args = parser.parse_args()

    res_dir = 'results'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # query files
    mol_file_smi = f"{res_dir}/{args.csv_name.split('.')[0]}.smi"
    collect_csv = f'{res_dir}/collect_filtered_{args.csv_name}'

    # 1.
    smis_list, smi2ids, mols_df = load_mols_csv(args.csv_name)
    with open(mol_file_smi, 'w') as f:
        f.writelines(smis_list)
    # 2
    if not os.path.exists(collect_csv):
        print("Read Original Data")
        original_mols_csv = f"{parent_path}/train/datasets/{args.original_mols_csv}.csv"
        print(f"csv: {original_mols_csv}")
        f_check = file_path = Path(original_mols_csv)
        assert f_check.exists(), f"{original_mols_csv} not exists"

        dict_mol_scaff, num_new_scaffs = extract_scaffolds(mol_file_smi, original_mols_csv)

        # 3 collect
        collect_filtered_mols_df = collect_mols_scaffs_props(mols_df, dict_mol_scaff)


        print(f"number of mols whose scaffold does not appear in the dataset: {num_new_scaffs}")

        collect_filtered_mols_df.to_csv(collect_csv, index=False)

    collect_filtered_mols_df = pd.read_csv(collect_csv)
    # print(collect_filtered_mols_df.shape[0])
    # assert 1 == 2

    if args.filter_pubchem:
        collect_filtered_mols_df, num_new_mols = filter_through_pubchem(collect_filtered_mols_df)
        print(f"number of mols not in the pubchem: {num_new_mols}")
        collect_filtered_mols_df.to_csv(collect_csv, index=False)

    png_dir = f"./example_pngs/{args.csv_name.split('.')[0]}"
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    visualize_mols_scaffs_props(collect_csv, png_dir)





