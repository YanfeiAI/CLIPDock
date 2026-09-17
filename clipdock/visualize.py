import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps

from clipdock.engine import score_only
from clipdock.scoring import GScorer, EScorer
from clipdock.utils.constant import GAUSS_PARAMS_PATH, DOCKING_PARAMS_PATH, DOCK_SCORE_K
from clipdock.utils.mol import protonate_mol


def save_lig_contrib_pdb(lig_mol, contrib, save_path):
    conformer = lig_mol.GetConformer()
    positions = conformer.GetPositions()
    atom_lines = []
    atom_serial_map = {}

    for i, atom in enumerate(lig_mol.GetAtoms()):
        pos = positions[i]
        serial_number = i + 1
        atom_symbol = atom.GetSymbol()
        atom_name = f"{atom_symbol}{serial_number}"
        if len(atom_name) > 4:
            atom_name = atom_name[:4]
        else:
            atom_name = atom_name.ljust(4)
        line = (
            f"ATOM  {serial_number:>5} {atom_name} LIG A   1    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            f"{1.0:6.2f}{contrib[i]:6.2f}          {atom_symbol:>2}  "
        )
        atom_lines.append(line)
        atom_serial_map[i] = serial_number

    connections = {}
    for bond in lig_mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        start_serial = atom_serial_map[start_idx]
        end_serial = atom_serial_map[end_idx]
        if start_serial not in connections:
            connections[start_serial] = []
        connections[start_serial].append(end_serial)

        if end_serial not in connections:
            connections[end_serial] = []
        connections[end_serial].append(start_serial)

    conect_lines = []
    for atom_serial, connected_atoms in connections.items():
        connected_atoms.sort()
        for i in range(0, len(connected_atoms), 4):
            group = connected_atoms[i:i + 4]
            conect_line = f"CONECT{atom_serial:>5}"
            for connected_serial in group:
                conect_line += f"{connected_serial:>5}"

            conect_lines.append(conect_line)
    pdb_content = "\n".join(atom_lines + conect_lines + ["END"])

    with open(save_path, 'w') as f:
        f.write(pdb_content)


def visualize_atom_contrib(name, lig_mol, ref_predictions, predictions, suffix=""):
    save_base = f"{name}{suffix}_contrib"
    png_path = f"{save_base}.png"
    pdb_path = f"{save_base}.pdb"

    atom_contrib = []
    relative_atom_contrib = []
    atom_num = lig_mol.GetNumAtoms()
    for i in range(len(predictions)):
        relative_delta = ref_predictions[0] / atom_num - predictions[i] / (atom_num - 1)
        relative_delta *= atom_num
        relative_atom_contrib.append(relative_delta)
        delta = ref_predictions[0] - predictions[i]
        atom_contrib.append(delta)

    draw_mol = Chem.Mol(lig_mol)
    for confId in range(draw_mol.GetNumConformers()):
        draw_mol.RemoveConformer(confId)
    AllChem.Compute2DCoords(draw_mol)

    template_mol = Chem.MolFromSmiles('COc1ccccc1OC')
    if draw_mol.HasSubstructMatch(template_mol):
        AllChem.Compute2DCoords(template_mol)
        AllChem.GenerateDepictionMatching2DStructure(draw_mol, template_mol)

    draw2d = Draw.MolDraw2DCairo(1024, 1024)
    SimilarityMaps.GetSimilarityMapFromWeights(draw_mol, relative_atom_contrib, draw2d=draw2d)
    draw2d.WriteDrawingText(png_path)

    save_lig_contrib_pdb(lig_mol, atom_contrib, pdb_path)


def atom_contrib(rec_mol, lig_mol, use_gauss=False, lig_pred_H=True,
                 gscore_params_path=GAUSS_PARAMS_PATH):
    gscore_params_path = gscore_params_path or GAUSS_PARAMS_PATH
    input_lig = Chem.Mol(lig_mol)
    rec_mol = Chem.RemoveAllHs(Chem.Mol(rec_mol))
    lig_mol = Chem.RemoveAllHs(Chem.Mol(lig_mol))
    rec_mol_h = protonate_mol(rec_mol)
    if lig_pred_H:
        lig_mol_h = protonate_mol(lig_mol)
    else:
        order = [a.GetIdx() for a in input_lig.GetAtoms() if a.GetAtomicNum() != 1]
        order += [a.GetIdx() for a in input_lig.GetAtoms() if a.GetAtomicNum() == 1]
        lig_mol_h = Chem.RenumberAtoms(input_lig, order)

    scorer = EScorer(rec_mol_h, lig_mol_h, DOCKING_PARAMS_PATH, 8, 8)
    score_list = [DOCK_SCORE_K *
                  (scorer.calc_escore(lig_mol.GetConformer().GetPositions(), lig_ignore_idx=i)
                   + scorer.rot_plt)
                  for i in range(lig_mol.GetNumAtoms())]
    if use_gauss:
        gauss_scorer = GScorer(rec_mol_h, lig_mol_h, gscore_params_path, 8, 8)
        ref_gauss = DOCK_SCORE_K * gauss_scorer.calc_gscore(lig_mol_h.GetConformer().GetPositions())
        gauss_score_list = [
            DOCK_SCORE_K * gauss_scorer.calc_gscore(lig_mol_h.GetConformer().GetPositions(), lig_ignore_idx=i)
            for i in range(lig_mol_h.GetNumAtoms())]
        for i in range(len(score_list)):
            atom = lig_mol_h.GetAtomWithIdx(i)
            score_list[i] += gauss_score_list[i]
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    score_list[i] += gauss_score_list[neighbor.GetIdx()] - ref_gauss
    return score_list


def analyse_scorer_atom_contrib(lig_mol, rec_mol, save_base_path, lig_pred_H=True,
                                gscore_params_path=GAUSS_PARAMS_PATH):
    gscore_params_path = gscore_params_path or GAUSS_PARAMS_PATH
    if lig_pred_H:
        lig_mol = protonate_mol(lig_mol)
    rec_mol = protonate_mol(rec_mol)
    name, ext = os.path.splitext(save_base_path)

    ref_predictions = score_only(rec_mol, [lig_mol], use_gauss=True, lig_pred_H=lig_pred_H,
                                 gscore_params_path=gscore_params_path)
    predictions = atom_contrib(rec_mol, lig_mol, use_gauss=True, lig_pred_H=lig_pred_H,
                               gscore_params_path=gscore_params_path)
    visualize_atom_contrib(name, Chem.RemoveAllHs(Chem.Mol(lig_mol)), ref_predictions, predictions, '')
