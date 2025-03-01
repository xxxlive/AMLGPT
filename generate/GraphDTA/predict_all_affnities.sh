#!/bin/bash

# 检查是否提供了 SMI_PATH 参数
if [ -z "$1" ]; then
  echo "error, pls check the arg"
  exit 1
fi

SMI_PATH=$1

# 定义 PROTEINS 数组
PROTEINS=('cdk2' 'EGFR' 'jak1' 'lrrk2' 'pim1' 'jak2')

# 遍历 PROTEINS 数组
# --use_original_file \
for p in "${PROTEINS[@]}"; do
  python predict_affniity.py \
    --use_original_file \
    --smiles_file "$SMI_PATH" \
    --fasta_path "protein_data/uniprotkb_$p.fasta"
done