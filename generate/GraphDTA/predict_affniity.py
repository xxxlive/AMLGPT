import argparse
import os
import sys
import time
import pandas as pd
import polars as pl
import torch
from torch_geometric.loader import DataLoader
import global_data
from tqdm import tqdm

from create_data_test import TestbedDataset, CustomTestbedDataset
from models.ginconv import GINConvNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
global_data._init()


from pathlib import Path

# 获取当前路径
current_path = Path.cwd()
parent_path = current_path.parent

print("cur_path:", current_path)
print("par_path:", parent_path)


TEST_BATCH_SIZE = 512

def predicting(model, device, loader):
    """
    It takes a model, a device, and a dataloader, and returns the predictions of the model on the data in the dataloader

    :param model: the model to be used for prediction
    :param device: the device to run the model on (CPU or GPU)
    :param loader: the dataloader for the test set
    :return: The predictions of the model.
    """
    model.eval()
    total_preds = torch.Tensor()
    # total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_preds.numpy().flatten()


def load_model_dict(model, ckpt):
    """
    It loads a model's state dictionary from a checkpoint file

    :param model: the model to load the weights into
    :param ckpt: the path to the checkpoint file
    """
    model.load_state_dict(torch.load(ckpt))


def main(args):
    """
        It loads the model, loads the test data, and then runs the model on the test data
        :param args: the arguments passed in from the command line
        """
    time_start = time.time()
    smiles_data = pd.read_csv(args.smiles_file)
    # 如果smiles_data不存在ID列，则添加ID列
    if 'ID' not in smiles_data.columns:
        smiles_data.insert(0, 'ID', range(len(smiles_data)), allow_duplicates=False)
    smiles_data.to_csv(args.smiles_file, index=False)

    data_df = pl.read_csv(args.smiles_file)
    file_name = os.path.splitext(os.path.basename(args.smiles_file))[0]
    print(file_name)

    root_path = args.data_path
    result_dir = 'results'
    test_data = CustomTestbedDataset(args=args, root=root_path, phase='test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    print('dataloader generate success')
    # training the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GINConvNet().to(device)
    load_model_dict(model, args.graph_model_path)
    pred = predicting(model, device, test_loader)

    pic50_mean = sum(pred) / len(pred)
    pic50_std = sum((x - pic50_mean) ** 2 for x in pred) / len(pred)

    time_end = time.time()
    time_use = (time_end - time_start) / 60
    print('test all time use : ', time_use)
    print("pic50_mean {}".format(pic50_mean))
    print("pic50_std {}".format(pic50_std))

    prot_name = args.fasta_path.split('.')[0].split('_')[-1]
    print(prot_name)
    mod_df = pd.read_csv(test_data.custom_smi_data)
    mod_df[prot_name] = pred
    print(mod_df.columns)
    # assert 1 == 2

    mod_df.to_csv(args.smiles_file, index=False)

    with open(args.record_file, 'a') as f:
        f.write(f"pic50_mean = {round(pic50_mean, 3)}, pic50_std = {round(pic50_std, 3)},\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add argument
    parser.add_argument('--use_original_file', action="store_true", help='whether use the original dataset')
    parser.add_argument('--smiles_file', default='collect_filtered_guaca_gen.csv', help='dataset ready to test')
    parser.add_argument('--fasta_path',type=str,default= 'protein_data/uniprotkb_cdk2.fasta',help='the path of protein fasta')
    parser.add_argument('--sep', type=str, default=',', help='the separator of CSV file')
    parser.add_argument('--smiles_col_name', type=str, default='SMILES', help='the column name of SMILES')
    parser.add_argument('--id_col_name', type=str, default='ID', help='the column name of id')
    parser.add_argument('--data_path', type=str, default='result/our_CDK2_valid',
                        help='the path of protein fasta')
    parser.add_argument('--record_file', type=str, default='', help='the filename that contains the pic50 results')

    # graph_model_path
    parser.add_argument('--graph_model_path', type=str, default='model/model_GINConvNet_ChEMBL_total_lrrk.model', help='checkpoint path of GraphDTA')
    args = parser.parse_args()

    args.record_file = f'result/{args.smiles_file.split(".")[0]}.txt'

    if args.use_original_file:
        args.smiles_file = f"/data/whb/Generation/AMLGPT/train/datasets/{args.smiles_file}"
    else:
        args.smiles_file = f"{parent_path}/{args.smiles_file}"


    # 长啥样
    global_data.set_fasta_path(args.fasta_path)
    main(args)

