from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from io import BytesIO
from PIL import Image
from cairosvg import svg2png

import os

# piC50 is high, qed is higher than 0.6

def generate_image(mol, highlight_atoms, highlight_bonds, atomColors, bondColors, radii, size, output, isNumber=False):
    print("Highlight Atoms:", highlight_atoms)
    print("Highlight Bonds:", highlight_bonds)
    print("Atom Colors:", atomColors)
    print("Bond Colors:", bondColors)

    image_data = BytesIO()
    view = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)

    option = view.drawOptions()
    if isNumber:
        for atom in mol.GetAtoms():
            option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1)

    view.DrawMolecule(tm,
                      highlightAtoms=highlight_atoms,
                      highlightBonds=highlight_bonds,
                      highlightAtomColors={atom: atomColors for atom in highlight_atoms},
                      highlightBondColors={bond: bondColors for bond in highlight_bonds},
                      highlightAtomRadii={atom: radii for atom in highlight_atoms})

    view.FinishDrawing()
    svg = view.GetDrawingText()
    SVG(svg.replace('svg:', ''))
    svg2png(bytestring=svg, write_to=output)
    img = Image.open(output)
    img.save(image_data, format='PNG')

    return image_data


def find_common_atoms_bonds(mol, submol):
    # common atoms
    comm_atoms = mol.GetSubstructMatches(submol)[0]
    # comm_bonds
    comm_bonds = set()

    # 获取与共同原子相连的边
    for atom_idx in comm_atoms:
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            comm_bonds.add(bond.GetIdx())

    return comm_atoms, comm_bonds


def vis_and_highlight_scaff(smi, scaff_smi, png_dir, cnt=1):
    """
    smi1: smiles of OMG
    smi2: smiles of moses
    """

    comm_frag = Chem.MolFromSmarts(scaff_smi)
    mol = Chem.MolFromSmiles(smi)

    comm_atoms, comm_bonds = find_common_atoms_bonds(mol, comm_frag)

    # Prepare colors
    atom_color = (0.95, 0.6, 0.0)
    bond_color = (0.95, 0.6, 0.0)
    radius = 0.3

    _ = generate_image(mol, list(comm_atoms), list(comm_bonds), atom_color, bond_color, radius,
                       (400, 400), f'{png_dir}/omg_mol_{cnt}.png', False)


# highest for cdk2, logp = 1.62
smi_1 = 'Cc1cccc2c1OP(=O)(OCC1OC(n3cc(C)c(=O)[nH]c3=O)CC1O)OC2'
scaff_1 = 'O=P1(OCC2CCCO2)OCc2ccccc2O1'

# highest for egfr, logp = 3.99
# smi_2 = 'Cc1ccc2c(=O)cc(-c3ccccn3)oc2c1'
# scaff_2 = 'O=c1ccoc(-c2ccccn2)c1'
# smi_2 = 'CCCCCc1cn(-c2ccc(C(=O)NC3CC(N=[N+]=[N-])C(CO)O3)cc2)nn1'
# scaff_2 = 'O=C(NC1CCCO1)c1ccc(-n2ccnn2)cc1'

# smi_2 = 'COc1ccc(-c2c(C#N)[n+]([O-])c3ccccc3[n+]2[O-])cc1'
# # c1ccccc1 succeed
# scaff_2_list = ['c1ccc(-c2c[nH+]c3ccccc3[nH+]2)cc1', 'c1ccc2[nH+]cc[nH+]c2c1', 'c1ccccc1', 'c1c[nH+]cc[nH+]1', 'c1ccc(-c2c[nH+]cc[nH+]2)cc1']

smi_2 = 'COCCNC(=O)c1ncn(-c2ccc(-c3ccccc3)cc2)c1C(=O)OC'
scaff_2 = 'c1ccc(-c2ccc(-n3ccnc3)cc2)cc1'

# highest for jak1, logp = 1.74
smi_3 = 'CS(=O)(=O)Nc1ccc(-c2cc(NCCN3CCOCC3)nc3n[nH]cc23)cc1'
scaff_3 = 'c1ccc(-c2ccnc3n[nH]cc23)cc1'

# highest for pim1, logp = 1.03
smi_4 = 'CC1CN(Cc2nc(-c3ncccn3)no2)CCN1c1ncccn1'
scaff_4 = 'c1cnc(-c2noc(CN3CCNCC3)n2)nc1'

# highest for lrrk2, logp = 1.04
smi_5 = 'CN(C)CC1C(=O)C=C(Nc2ccc3c(c2)OCO3)C1=O'
scaff_5 = 'O=C1C=C(Nc2ccc3c(c2)OCO3)C(=O)C1'

# highest for qed, logp = 1.99
smi_6 = 'COc1nc2c(c(N)c1S(=O)(=O)c1ccccc1)CCC2'
scaff_6 = 'O=S(=O)(c1ccccc1)c1cnc2c(c1)CCC2'
# back up highest qed, lowest sas
smi_7 = 'COc1ccccc1NC(=O)c1ccc(=O)n(C2CCCCC2)c1'
scaff_7 = 'O=c1ccccn1C1CCCCC1'

png_dir = 'multi_props_pareto'
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

vis_and_highlight_scaff(smi_1, scaff_1, png_dir, 1)
# for scaf in scaff_2_list:
#     try:
#         vis_and_highlight_scaff(smi_2, scaf, png_dir, 2)
#         print(f"scaff {scaf} succeed")
#         # break  # 如果函数成功执行，跳出循环
#     except Exception as e:
#         # print(f"Error with scaffolding {scaf}: {e}. Trying next scaffold...")
#         continue  # 如果遇到异常，跳过当前scaf并尝试下一个
vis_and_highlight_scaff(smi_2, scaff_2, png_dir, 2)
vis_and_highlight_scaff(smi_3, scaff_3, png_dir, 3)
vis_and_highlight_scaff(smi_4, scaff_4, png_dir, 4)
vis_and_highlight_scaff(smi_5, scaff_5, png_dir, 5)
vis_and_highlight_scaff(smi_6, scaff_6, png_dir, 6)
vis_and_highlight_scaff(smi_7, scaff_7, png_dir, 7)