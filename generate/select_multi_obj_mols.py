import pandas as pd
import argparse
import os
import numpy as np


def select_multi_obj_mols(novel_df, original_df):
    """
    select molecules that achieved better properties in multiple aspect
    return : df
    """

    original_df['neg_sas'] = -original_df['sas']
    novel_df['neg_sas'] = -novel_df['sas']

    novel_df = novel_df.merge(original_df[['smiles', 'cdk2', 'egfr', 'jak1', 'lrrk2', 'pim1']], on='smiles', how='left')

    # 统计 original_df 中 'qed', 'logp', 'sas' 的范围
    qed_min, qed_max = original_df['qed'].min(), original_df['qed'].max()
    # logp_min, logp_max = original_df['logp'].min(), original_df['logp'].max()
    sas_min, sas_max = original_df['neg_sas'].min(), original_df['neg_sas'].max()
    cdk2_min, cdk2_max = original_df['cdk2'].min(), original_df['cdk2'].max()
    egfr_min, egfr_max = original_df['egfr'].min(), original_df['egfr'].max()
    jak1_min, jak1_max = original_df['jak1'].min(), original_df['jak1'].max()
    lrrk2_min, lrrk2_max = original_df['lrrk2'].min(), original_df['lrrk2'].max()
    pim1_min, pim1_max = original_df['pim1'].min(), original_df['pim1'].max()


    # 过滤 novel_df，确保值在 original_df 的范围内
    # (novel_df['logp'] >= logp_min) & (novel_df['logp'] <= logp_max) &
    filtered_novel_df = novel_df[
        (novel_df['qed'] >= qed_min) & (novel_df['qed'] <= qed_max) &
        (novel_df['neg_sas'] >= sas_min) & (novel_df['neg_sas'] <= sas_max) &
        (novel_df['cdk2'] >= cdk2_min) & (novel_df['cdk2'] <= cdk2_max) &
        (novel_df['egfr'] >= egfr_min) & (novel_df['egfr'] <= egfr_max) &
        (novel_df['jak1'] >= jak1_min) & (novel_df['jak1'] <= jak1_max) &
        (novel_df['lrrk2'] >= lrrk2_min) & (novel_df['lrrk2'] <= lrrk2_max) &
        (novel_df['pim1'] >= pim1_min) & (novel_df['pim1'] <= pim1_max)
        ]

    def is_pareto_efficient(costs):
        """
        找到帕累托前沿的点。
        :param costs: 一个 (n_points, n_costs) 的数组
        :return: 一个 (n_points, ) 的布尔数组，表示每个点是否在帕累托前沿
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # 将所有被 c 支配的点标记为非帕累托前沿
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True  # 保持当前点
        return is_efficient

    # 提取 filtered_novel_df 中的目标列
    targets = filtered_novel_df[['qed', 'neg_sas', 'cdk2', 'egfr', 'jak1', 'lrrk2', 'pim1']].values

    # 由于 qed, logp, sas 都是越大越好，我们需要取反以适配帕累托前沿函数
    costs = -targets

    # 找到帕累托前沿的点
    pareto_mask = is_pareto_efficient(costs)

    # 获取帕累托前沿的子集
    pareto_df = filtered_novel_df[pareto_mask]

    # mean_values = original_df[['qed', 'logp', 'neg_sas']].mean()
    #
    # print("Mean values from original_df:")
    # print(mean_values)
    #
    # # 筛选 novel_df 中 qed, logp 和 sas 都大于上面均值的行
    # filtered_novel_df = novel_df[
    #     (novel_df['qed'] > mean_values['qed']) &
    #     (novel_df['logp'] > mean_values['logp']) &
    #     (novel_df['sas'] < mean_values['sas'])
    #     ]
    return pareto_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['guaca', 'moses'], default='guaca', help='dataset path')
    args = parser.parse_args()

    noval_scaff_df = pd.read_csv(f'results/collect_filtered_{args.data_name}_gen.csv')
    original_df = pd.read_csv(f'{args.data_name}_gen.csv')

    noval_scaff_df.columns = noval_scaff_df.columns.str.lower()
    original_df.columns = original_df.columns.str.lower()

    selected_df = select_multi_obj_mols(noval_scaff_df, original_df)
    save_dir = 'selected_res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    selected_df.to_csv(f'{save_dir}/selected_{args.data_name}_gen.csv', index=False)