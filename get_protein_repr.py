import argparse
import glob
import json
import lmdb
import numpy as np
import os
import pandas as pd
import pickle
import re
from biopandas.pdb import PandasPdb
from tqdm import tqdm

main_atoms = ["N", "CA", "C", "O", "H"]

def load_from_CASF(pdb_id, protein_path):
    try:
        pdb_path = os.path.join(protein_path, 'protein', pdb_id + "_protein.pdb")
        pmol = PandasPdb().read_pdb(pdb_path)
        pocket_residues = json.load(
            open(os.path.join(protein_path, "protein.pocket.json"))
        )[pdb_id]
        return pmol, pocket_residues
    except:
        print("Currently not support parsing pdb and pocket info from local files.")

def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def parser_pdb(pdb_id, protein_path):
    pmol, pocket_residues = load_from_CASF(pdb_id, protein_path)
    
    pname = pdb_id
    pro_atom = pmol.df["ATOM"]
    pro_hetatm = pmol.df["HETATM"]

    pro_atom["ID"] = pro_atom["chain_id"].astype(str) + pro_atom[
        "residue_number"
    ].astype(str)
    pro_hetatm["ID"] = pro_hetatm["chain_id"].astype(str) + pro_hetatm[
        "residue_number"
    ].astype(str)

    pocket = pd.concat(
        [
            pro_atom[pro_atom["ID"].isin(pocket_residues)],
            pro_hetatm[pro_hetatm["ID"].isin(pocket_residues)],
        ],
        axis=0,
        ignore_index=True,
    )
    # print(pocket)
    pocket["normalize_atom"] = pocket["atom_name"].map(normalize_atoms)
    pocket = pocket[pocket["normalize_atom"] != ""]
    patoms = pocket["atom_name"].apply(normalize_atoms).values.tolist()
    # print(patoms, len(patoms))
    pcoords = [pocket[["x_coord", "y_coord", "z_coord"]].values]
    side = [0 if a in main_atoms else 1 for a in patoms]
    residues = (
        pocket["chain_id"].astype(str) + pocket["residue_number"].astype(str)
    ).values.tolist()

    return pickle.dumps(
        {
            "atoms": patoms,
            "coordinates": pcoords,
            "side": side,
            "residue": residues,
            "pdbid": pname,
        },
        protocol=-1,
    )

def write_lmdb(pdb_id_list, job_name, outpath="./results", protein_path=""):
    os.makedirs(outpath, exist_ok=True)
    outputfilename = os.path.join(outpath, job_name + ".lmdb")
    try:
        os.remove(outputfilename)
    except:
        pass
    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn_write = env_new.begin(write=True)
    for i, pdb_id in tqdm(enumerate(pdb_id_list)):
        inner_output = parser_pdb(pdb_id, protein_path)
        txn_write.put(f"{i}".encode("ascii"), inner_output)
    txn_write.commit()
    env_new.close()

def get_csv_results(predict_path, results_path):
    predict = pd.read_pickle(predict_path)
    pdb_id_list, mol_repr_list, atom_repr_list, pair_repr_list = [], [], [], []
    for batch in predict:
        sz = batch["bsz"]
        new_dict = dict()
        for i in range(sz):
            pdb_id_list.append(batch["data_name"][i])
            mol_repr_list.append(batch["mol_repr_cls"][i])
            atom_repr_list.append(batch['atom_repr'][i])
            pair_repr_list.append(batch["pair_repr"][i])

            new_dict["data_name"] = batch["data_name"][i]
            new_dict["mol_repr_cls"] = batch["mol_repr_cls"][i]
            new_dict["atom_repr"] = batch["atom_repr"][i]
            new_dict["pair_repr"] = batch["pair_repr"][i]
            
            np.save('{}/{}.npy'.format(results_path, batch["data_name"][i]), new_dict)
            
            print(batch["mol_repr_cls"][i].shape, batch['atom_repr'][i].shape, batch["pair_repr"][i].shape)
    predict_df = pd.DataFrame({"pdb_id": pdb_id_list, "mol_repr": mol_repr_list, "atom_repr": atom_repr_list, "pair_repr": pair_repr_list})
    print(predict_df.head(1),predict_df.info())
    predict_df.to_csv(results_path+'/mol_repr.csv',index=False)

def main(args):
    only_polar = 0  # no h
    dict_name = 'dict_coarse.txt'
    batch_size = 2
    
    results_path= args.results_path   # replace to your save path
    casf_collect = os.listdir(os.path.join(args.data, "protein"))
    casf_collect = list(set([item[:4] for item in casf_collect]))
    write_lmdb(casf_collect, job_name=args.job_name, outpath=results_path, protein_path=args.data)
    
    # NOTE: Currently, the inference is only supported to run on a single GPU. You can add CUDA_VISIBLE_DEVICES="0" before the command.
    os.system("cp ./Unimol/unimol/example_data/pocket/{} {}".format(dict_name, results_path))
    os.system('CUDA_VISIBLE_DEVICES="0" python ./Unimol/unimol/unimol/infer.py --user-dir ./Unimol/unimol/unimol {} --valid-subset {} --results-path {} --num-workers 1 --ddp-backend=c10d --batch-size {} --task unimol_pocket --loss unimol_infer --arch unimol_base --path {} --dict-name {} --log-interval 50 --log-format simple --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer'.format(results_path, args.job_name, results_path, batch_size, args.weight_path, dict_name))
    
    pkl_path = glob.glob(f'{results_path}/*_{args.job_name}.out.pkl')[0]
    get_csv_results(pkl_path, results_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./Data/PDB", help='path of proteins')
    parser.add_argument('--job_name', type=str, default='get_pocket_repr', help='job_name')
    parser.add_argument('--results_path', type=str, default='./Data/PDB/repr', help='path to save results')
    parser.add_argument('--weight_path', type=str, default='./Unimol/unimol_tools/unimol_tools/weights/pocket_pre_220816.pt', help='path of Unimol checkpoints')

    args = parser.parse_args()

    main(args)