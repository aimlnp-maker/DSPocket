import argparse
import os

from model.DSPocket import MolTrain

def main(args):
    data = "{}/Train_data_{}.csv".format(args.data, args.task)
    save_dir = "{}/{}".format(args.save_path, args.task)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clf = MolTrain(task=args.task,
                    data_type=args.data_type,
                    save_path=save_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    protein_path=args.protein_path,
                    )

    pred = clf.fit(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="", help='path of training data')
    parser.add_argument('--data_type', type=str, default='molecule', help='data type')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default="regression", help='task_type')
    parser.add_argument('--epochs', type=int, default=30, help='epoch of training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--protein_path', type=str, default="./Data/PDB/repr", help='path of model')
    parser.add_argument('--save_path', type=str, default="./Checkpoints", help='path of model')

    args = parser.parse_args()

    main(args)