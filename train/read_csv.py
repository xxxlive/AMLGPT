import pandas as pd
import re


moses_df = pd.read_csv('datasets/moses2.csv')
guaca_df = pd.read_csv('datasets/guacamol2.csv')
jak_df = pd.read_csv('datasets/train_val_JAK2.csv')

# 使用merge根据'smiles'列进行合并，找到共同的值
common_smiles = pd.merge(guaca_df[['SMILES']], jak_df[['SMILES']], on='SMILES', how='inner')

# 查看共有的smiles值
print(len(common_smiles))

filtered_jak_df = jak_df[~jak_df['SMILES'].isin(common_smiles['SMILES'])]

# tokenize
pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)

smiles = filtered_jak_df['SMILES']
lens = [len(regex.findall(i.strip())) for i in (list(smiles.values))]

# 将lens转换为Series
lens_series = pd.Series(lens)

# 过滤出train_data中lens>100的行
filtered_jak_df = filtered_jak_df[lens_series[:len(filtered_jak_df)] < 100]


# 重置索引
filtered_jak_df.reset_index(drop=True, inplace=True)

# 保存为新的 CSV 文件
filtered_jak_df.to_csv('datasets/train_val_JAK2.csv', index=False)

# guaca_df['jak2'] = 0

print(moses_df.head(3))
print(guaca_df.head(3))

# guaca_df.to_csv('guacamol2.csv', index=False)