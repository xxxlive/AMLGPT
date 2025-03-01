import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Contrib.SA_Score import sascorer

import os

# 计算分子的SAscore和LogP
def add_props(mols_df):
    sas_list = []
    logp_list = []
    for smi in mols_df['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol:  # 确保分子有效
            sas_list.append(sascorer.calculateScore(mol))
            logp_list.append(Crippen.MolLogP(mol))
        else:
            sas_list.append(None)  # 如果SMILES无效，填充None
            logp_list.append(None)

    mols_df['sas'] = sas_list
    mols_df['logp'] = logp_list
    return mols_df


def plot_kde_comparison(mol_df_1, mol_df_2, mol_df_3, prop):
    # 提取三个DataFrame的属性列
    logp_jak2 = mol_df_1[prop]
    logp_guaca_gen = mol_df_2[prop]
    logp_guaca_jak2_gen = mol_df_3[prop]

    # 绘制KDE分布曲线图
    plt.figure(figsize=(12, 8))
    sns.kdeplot(logp_jak2, label='JAK2 Original', fill=True, alpha=0.5, linewidth=2)     # , color='blue'
    sns.kdeplot(logp_guaca_gen, label='OMG-GPT (Guacamol)', fill=True, alpha=0.5, linewidth=2)   # , color='green'
    sns.kdeplot(logp_guaca_jak2_gen, label='OMG-GPT (Guacamol+JAK2)', fill=True, alpha=0.5, linewidth=2)     # , color='red'

    # 设置图形标题和标签
    # plt.title(f'KDE Distribution of {prop}')

    xlabel_name_dict = {'logp':'LogP', 'qed':'QED', 'sas':'SAS'}

    plt.xlabel(xlabel_name_dict[prop], fontsize=20)
    plt.ylabel('Density',fontsize=20)

    # 调整坐标轴刻度数字的字体大小
    plt.tick_params(axis='both', labelsize=14)

    plt.grid(True)

    # 显示图例
    plt.legend(fontsize=16)

    save_dir = 'prop_kde_curve_comparisons'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 显示图表
    plt.savefig(os.path.join(save_dir, f'{prop}-comp.png'))


# 读取数据
jak2_original_df = pd.read_csv('train_val_JAK2_filtered.csv')
guaca_gen_df = pd.read_csv('22_guaca_gen.csv')
guaca_jak2_gen_df = pd.read_csv('gua_jak_gen.csv')

jak2_original_df.columns = jak2_original_df.columns.str.lower()
guaca_gen_df.columns = guaca_gen_df.columns.str.lower()
guaca_jak2_gen_df.columns = guaca_jak2_gen_df.columns.str.lower()

# 添加SAscore和LogP列
jak2_original_df = add_props(jak2_original_df)
# guaca_gen_df = add_props(guaca_gen_df)
# guaca_jak2_gen_df = add_props(guaca_jak2_gen_df)

plot_kde_comparison(jak2_original_df, guaca_gen_df, guaca_jak2_gen_df, prop='logp')
plot_kde_comparison(jak2_original_df, guaca_gen_df, guaca_jak2_gen_df, prop='qed')
plot_kde_comparison(jak2_original_df, guaca_gen_df, guaca_jak2_gen_df, prop='sas')



# import pandas as pd
# from rdkit import Chem
#
# from rdkit.Chem import Crippen
# from rdkit.Contrib.SA_Score import sascorer
#
# # matplotlib - kde plot
#
# def add_props(mols_df):
#     sas_list = []
#     logp_list = []
#     for smi in mols_df['smiles']:
#         mol = Chem.MolFromSmiles(smi)
#         sas_list.append(sascorer.calculateScore(mol))
#         logp_list.append(Crippen.MolLogP(mol))
#         # 将计算的SAscore和LogP值添加为新列
#     mols_df['sas'] = sas_list
#     mols_df['logp'] = logp_list
#
#     return mols_df
#
# jak2_original_df = pd.read_csv('train_val_JAK2_filtered.csv')
# guaca_gen_df = pd.read_csv('22_guaca_gen.csv')
# guaca_jak2_gen_df = pd.read_csv('gua_jak_gen.csv')
#
# jak2_original_df.columns = jak2_original_df.columns.str.lower()
# guaca_gen_df.columns = guaca_gen_df.columns.str.lower()
# guaca_jak2_gen_df.columns = guaca_jak2_gen_df.columns.str.lower()
#
# jak2_original_df = add_props(jak2_original_df)





