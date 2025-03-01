# Implementation of OMG Framework: Online Knowledge Distillation for Molecule Generation


## Conda Environment Preparation
Use the following command lines to create the conda environment for molecular generation experiment.   
The relying packages are recorded in the requirement.txt
```
conda create -n mole_gen_omg python==3.9
conda activate mole_gen_omg
pip install -r requirements.txt
```

## Data Preparation
### Links to Original DataSets 
 Original Guacamol dataset can be found here:

https://github.com/BenevolentAI/guacamol

- Original Moses dataset can be found here:

https://github.com/molecularsets/moses

The preprocessed datasets are stored in .zip file, users should enter the path of the datasets and unzip them first.
```
cd train/datasets
unzip dataset.zip
```

## 1. Train models in the OMG framework
### 1.1 Train models from scratch
We offered an exemplary command line to train the molecular generation models, with recommended hyperparameters.
```python
cd train                            # enter into the directory for training

python train.py
--run_name 2025-2-11-omg-moses      # "xxx" is any customized strings, used to distinguish each experiment
--data_name  moses2                 # options: [moses2, guacamol2] -- the dataset used, moses or guacamol      
--l1 1.5                            # lambda_1, weight of cross-entropy loss for GPT part (unconditional generation)
--l2 1                              # lambda_2, weight of cross-entropy loss for scaffold-based Transformer part
--l3 0.5                            # lambda_3, weight of kl-divergence of mutual-learning mechanism
--t 1.0                             # temperature in softmax function when predicting the SMILES tokens
                                    # (As t becomes larger, the distribution of each SMILES token becomes more uniform)
--max_epoch 100                     # total epochs for training the models in the OMG framework
```

User can directly copy the following command line:
```angular2html
python train.py --run_name 2025-2-11-omg-guacamol --data_name guacamol2 --l1 1.5 --l2 1 --l3 0.5 --l3 0.5 --t 1.0 --max_epoch 100
```

### 1.1 Fine-tune models
The trained models on the MOSES or Guacamol dataset can be further trained on other dataset, we construct a in-house datasets consists of molecules have high binding affinity with JAK2 protein, referred to the dataset from [LMLF](https://github.com/Shreyas-Bhat/LMLF/blob/main/SMILES/JAK2.txt).
The fine-tune command is as follows:
```angular2html
python fineTune.py --run_name 2025-2-11-omg-guacamol-jak2 --data_name train_val_JAK2_filtered --l1 1.5 --l2 1 --l3 0.5 --l3 0.5 --t 1.0 --max_epoch 100 --model_weight ../cond_gpt/weights/2025-2-11-omg-guacamol.ptgen.pt --model_weight2 ../cond_gpt/weights/2025-2-11-omg-guacamol.ptscaffold.pt
```
The weights of train or fine-tuned models are stored in directory 'OMG-GPT/cond_gpt/weights'.


## 2. Perform Unconditional Molecule Generation
The command line to generate SMILEs strings of molecules from scratch is listed below.
```python
cd generate
python generate.py 
--model_weight cond_gpt/weights/2025-2-11-omg-moses.ptgen.pt        # path to weights of pretrained models
--csv_name omg-moses-gen                                            # name of the csv file that contains the generation results
--gen_size  1000                                                    # number of molecules to be generated, as you want
--vocab_size  26                                                    # number of SMILES tokens used to generate molecules belong to the ehecmical space of selected dataset
                                                                    # moses: 26, guacamol: 95
--block_size 54                                                     # max lengths of SMILES strings to be generated 
                                                                    # moses: 54, guacamol: 100
```
User can directly copy the following command line:
```
python generate.py --model_weight cond_gpt/weights/2025-2-11-omg-guacamol.ptgen.pt --csv_name omg-guacamol-gen --gen_size 1000 --vocab_size 95 --block_size 100
```
The csv file 'omg-guacamol-gen.csv' contains SMILES strings and corresponding chemical property values including LogP, QED and SAS.

## 3. Other Options
### 3.1 Evaluate Biological Activity
We use [GraphDTA](https://github.com/ecust-hc/ScaffoldGVAE/tree/master/GraphDTA) to evaluate the pIC50 scores of the generated molecules against five proteins, including CDK2, EGFR, JAK1, LRRK2, and PIM1, by running the bash script below:
```
cd generate/GraphDTA
sh predict_all_affnities.sh omg-guacamol-gen
```
Please note that the argument following can be replaced by any 'csv_name' input in the command line of 'generate.py'.
The results of mean and std values of pIC50 are stored in the file 'results/omg-guacamol-gen.txt'.

### 3.2 Collect Novel Molecules or Novel Scaffolds

We can also collect molecules whose scaffolds are not presented in the existing datasets.
For Guacamol dataset:
```
python extract_filter_novel_scaf.py --csv_name omg-guacamol-gen --original_mols_csv guacamol2
```
For MOSES dataset:
```
python extract_filter_novel_scaf.py --csv_name omg-moses-gen --original_mols_csv moses2
```
To further filter out the molecules not presented in Pubchem database, one can add argument '--filter_pubchem' in the tail of the command line.

