import argparse
import os
import pandas as pd

from convert import convert_results
from model.DSPocket import MolPredict

def main(args):
    tasks = ['classification', 'regression']

    for task in tasks:
        model_path = os.path.join(args.model, task)
        clf = MolPredict(load_model=model_path, protein_path=args.protein_path)
        test_pred = clf.predict(data=args.data)
        test_results = pd.DataFrame({'pred':test_pred.flatten(),
                                   'smiles':clf.datahub.data['smiles']
                                    })

        if not os.path.exists(args.out):
            os.makedirs(args.out)

        test_results.to_csv("{}/{}_results.csv".format(args.out, task), index=False, sep=",")

    convert_results("{}/{}_results.csv".format(args.out, "classification"), "{}/{}_results.csv".format(args.out, "regression"), args.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="", help='dir of model checkpoint')
    parser.add_argument('--data', type=str, default="", help='path of data')
    parser.add_argument('--out', type=str, default="./result", help='path of output')
    parser.add_argument('--protein_path', type=str, default="./Data/PDB/repr", help='path of model')
    args = parser.parse_args()

    main(args)
