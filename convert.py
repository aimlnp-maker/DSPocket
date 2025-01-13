import argparse
import pandas as pd

def convert_results(classification_data, regression_data, melocular_lib):
    test_classfication_data = pd.read_csv(classification_data)
    test_regression_data = pd.read_csv(regression_data)

    all_lib_data = pd.read_csv(melocular_lib)

    all_classfication_test_data = test_classfication_data['pred'].tolist()
    all_classfication_test_smiles = test_classfication_data['smiles'].tolist()

    all_regression_test_data = test_regression_data['pred'].tolist()
    all_regression_test_smiles = test_regression_data['smiles'].tolist()

    all_lib_smiles = all_lib_data['SMILES'].tolist()
    all_ID = all_lib_data['ID'].tolist()
    all_STAND_ID = all_lib_data['STAND_ID'].tolist()
    all_PROTEIN = all_lib_data['PROTEIN'].tolist()
    all_LABEL = all_lib_data['LABEL'].tolist()

    new_classfication_ID, new_classfication_STAND_ID, new_classfication_PROTEIN, new_classfication_LABEL = [], [], [], []
    for index, i in enumerate(all_classfication_test_smiles):
        t_index = all_lib_smiles.index(i)
        new_classfication_ID.append(all_ID[t_index])
        new_classfication_STAND_ID.append(all_STAND_ID[t_index])
        new_classfication_PROTEIN.append(all_PROTEIN[t_index])
        new_classfication_LABEL.append(all_LABEL[t_index])

    new_regression_ID, new_regression_STAND_ID, new_regression_PROTEIN, new_regression_LABEL = [], [], [], []
    for index, i in enumerate(all_regression_test_smiles):
        t_index = all_lib_smiles.index(i)
        new_regression_ID.append(all_ID[t_index])
        new_regression_STAND_ID.append(all_STAND_ID[t_index])
        new_regression_PROTEIN.append(all_PROTEIN[t_index])
        new_regression_LABEL.append(all_LABEL[t_index])

    data_classfication = pd.DataFrame({'pred': all_classfication_test_data, 'smiles': all_classfication_test_smiles, 'ID': new_classfication_ID, 'STAND_ID': new_classfication_STAND_ID, 'PROTEIN': new_classfication_PROTEIN, 'LABEL': new_classfication_LABEL})
    data_classfication.to_csv(classification_data)

    data_regression = pd.DataFrame({'pred': all_regression_test_data, 'smiles': all_regression_test_smiles, 'ID': new_regression_ID, 'STAND_ID': new_regression_STAND_ID, 'PROTEIN': new_regression_PROTEIN, 'LABEL': new_regression_LABEL})
    data_regression.to_csv(regression_data)