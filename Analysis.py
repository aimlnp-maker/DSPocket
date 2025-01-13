import argparse
import pandas as pd

def get_classficatin_pred(similes, classfication_smiles, classfication_prob):
    s_index = classfication_smiles.index(similes)
    pred = classfication_prob[s_index]
    return pred

def analysis_LABEL(LABEL):
    circle_list = {'C{}'.format(i): 0 for i in range(15)}
    Linear_list = {'L{}'.format(i): 0 for i in range(4)}
    # print(circle_list, Linear_list)

    for label in LABEL:
        t_l = label.split("_")
        if len(t_l) == 1:
            Linear_list[t_l[0]] = Linear_list[t_l[0]] + 1
        elif len(t_l) == 3:
            Linear_list[t_l[0]] = Linear_list[t_l[0]] + 1
            circle_list[t_l[1]] = circle_list[t_l[1]] + 1
        else:
            print("unknow type!!")
    
    return Linear_list, circle_list

def analysis_ID(ID):
    a_list = {'{}a{}'.format(i, j): 0 for i in range(1, 5) for j in range(1, 31)}
    b_list = {'{}b{}'.format(i, j): 0 for i in range(1, 5) for j in range(1, 15)}
    # print(a_list)
    # print(b_list)
    for id in ID:
        if type(id) == str:
            if 'a' in id and 'b' in id:
                b_index = id.index('b')
                # print(id, id[:(b_index-1)], id[(b_index-1):])
                a_list[id[:(b_index-1)]] = a_list[id[:(b_index-1)]] + 1
                b_list[id[(b_index-1):]] = b_list[id[(b_index-1):]] + 1
            else:
                if id in a_list.keys():
                    a_list[id] = a_list[id] + 1
                elif id in b_list.keys():
                    b_list[id] = b_list[id] + 1

    return a_list, b_list

def analysis_STAND_ID(STAND_ID):
    A_list = {'A{}'.format(i): 0 for i in range(0, 417)}
    B_list = {'B{}'.format(i): 0 for i in range(0, 414)}
    for id in STAND_ID:
        if type(id) == str:
            B_index = id.index('B')

            A_list[id[:B_index]] = A_list[id[:B_index]] + 1
            B_list[id[B_index:]] = B_list[id[B_index:]] + 1

    return A_list, B_list

def analysis_data(ID, LABEL, STAND_ID):

    a_list, b_list = analysis_ID(ID)
    Linear_list, circle_list = analysis_LABEL(LABEL)
    A_list, B_list = analysis_STAND_ID(STAND_ID)

    return a_list, b_list, Linear_list, circle_list, A_list, B_list

def cal_confidence(ref_dict, c_dict):
    confidence_dict = {}
    for key in ref_dict.keys():
        ref_count = ref_dict[key]
        c_count = c_dict[key]

        if ref_count == 0:
            confidence_dict[key] = 0
            continue
        else:
            confidence_dict[key] = c_count / ref_count
    
    return confidence_dict

def get_confidence(a_list, b_list, Linear_list, circle_list, A_list, B_list, regression_ID, regression_LABEL, regression_STAND_ID):
    ref_a_list, ref_b_list = analysis_ID(regression_ID)
    ref_linear_list, ref_circle_list = analysis_LABEL(regression_LABEL)
    ref_A_list, ref_B_list = analysis_STAND_ID(regression_STAND_ID)

    a_confidence = cal_confidence(ref_a_list, a_list)
    b_confidence = cal_confidence(ref_b_list, b_list)
    linear_confidence = cal_confidence(ref_linear_list, Linear_list)
    circle_confidence = cal_confidence(ref_circle_list, circle_list)
    A_confidence = cal_confidence(ref_A_list, A_list)
    B_confidence = cal_confidence(ref_B_list, B_list)
    return a_confidence, b_confidence, linear_confidence, circle_confidence, A_confidence, B_confidence

def show_confidence(counts_dict, confidence_dict, num=10, ID_2_STANDID_DICT=None):

    keys = counts_dict.keys()
    counts = counts_dict.values()
    confidences = confidence_dict.values()

    df = pd.DataFrame({'name': keys, 'count': counts, 'confidence': confidences})
    sort_df = df.sort_values(by='confidence', ascending=False)

    top_keys = sort_df['name'].tolist()[: num]
    top_counts = sort_df['count'].tolist()[: num]
    top_confidences = sort_df['confidence'].tolist()[: num]

    if ID_2_STANDID_DICT:
        for n_idx, n in enumerate(top_keys):
            T_ID = ID_2_STANDID_DICT[n]
            if type(T_ID) == str:
                top_keys[n_idx] = n + "({})".format(T_ID)

    show_keys, show_counts, show_confidences = ['name'], ['count'], ['confidence']
    show_keys.extend(top_keys)
    show_counts.extend(top_counts)
    show_confidences.extend(top_confidences)

    show_df = pd.DataFrame([show_keys, show_counts, show_confidences])
    print(show_df.to_string(index=False, header=False))
    return None

def main(args):
    classfication_results = args.classfication_results
    regression_results = args.regression_results
    result_path = "{}/result.csv".format(args.save)
    cutoff = args.cutoff

    classfication_data = pd.read_csv(classfication_results)
    Regression_data = pd.read_csv(regression_results)

    sort_classfication_data = classfication_data.sort_values(by='pred', ascending=False)
    sort_regression_data = Regression_data.sort_values(by='pred', ascending=True)

    # classfication
    classfication_prob = sort_classfication_data['pred'].tolist()
    classfication_smiles = sort_classfication_data['smiles'].tolist()
    classfication_ID = sort_classfication_data['ID'].tolist()
    classfication_STAND_ID = sort_classfication_data['STAND_ID'].tolist()
    classfication_LABEL = sort_classfication_data['LABEL'].tolist()

    # regression
    regression_pred = sort_regression_data['pred'].tolist()
    regression_smiles = sort_regression_data['smiles'].tolist()
    regression_ID = sort_regression_data['ID'].tolist()
    regression_STAND_ID = sort_regression_data['STAND_ID'].tolist()
    regression_LABEL = sort_regression_data['LABEL'].tolist()

    A_ID_2_STANDID_DICT, B_ID_2_STANDID_DICT = {}, {}
    A_ref_csv = pd.read_csv('Data/a_stand_lib.tsv', sep='\t')
    B_ref_csv = pd.read_csv('Data/s_stand_lib.tsv', sep='\t')

    A_ID = A_ref_csv['name'].tolist()
    A_STAND_ID = A_ref_csv['stand_name'].tolist()
    B_ID = B_ref_csv['name'].tolist()
    B_STAND_ID = B_ref_csv['stand_name'].tolist()

    for t_id, stand_id in zip(A_ID, A_STAND_ID):
        A_ID_2_STANDID_DICT[stand_id] = t_id

    for t_id, stand_id in zip(B_ID, B_STAND_ID):
        B_ID_2_STANDID_DICT[stand_id] = t_id

    print(len(classfication_smiles), len(regression_smiles))

    t_regression = []
    t_regression_pred = []
    t_regression_ID = []
    t_regression_STAND_ID = []
    t_regression_LABEL = []
    for t_p, t_s in zip(regression_pred, regression_smiles):
        t_c_p = get_classficatin_pred(t_s, classfication_smiles, classfication_prob)
        if float(t_c_p) < 0.8:
            continue
        else:
            t_regression.append(t_s)
            t_regression_pred.append(t_p)
            t_regression_ID.append(regression_ID[regression_smiles.index(t_s)])
            t_regression_STAND_ID.append(regression_STAND_ID[regression_smiles.index(t_s)])
            t_regression_LABEL.append(regression_LABEL[regression_smiles.index(t_s)])

    print(len(t_regression))

    if os.path.exist(result_path):
        df = pd.DataFrame({'smiles': t_regression, 'pred': t_regression_pred, 'ID': t_regression_ID,
                           'STAND_ID': t_regression_STAND_ID, 'LABEL': t_regression_LABEL})
        df.to_csv(result_path, sep='\t')

    a_list, b_list, Linear_list, circle_list, A_list, B_list = analysis_data(t_regression_ID[:cutoff],
                                                                             t_regression_LABEL[:cutoff],
                                                                             t_regression_STAND_ID[:cutoff])
    a_confidence, b_confidence, linear_confidence, circle_confidence, A_confidence, B_confidence = get_confidence(
        a_list, b_list, Linear_list, circle_list, A_list, B_list, regression_ID, regression_LABEL, regression_STAND_ID)

    print('*' * 45 + '  Analysis a ' + '*' * 45)
    show_confidence(a_list, a_confidence, 20)

    print('*' * 45 + '  Analysis b ' + '*' * 45)
    show_confidence(b_list, b_confidence, 10)

    print('*' * 45 + '  Analysis Linear ' + '*' * 45)
    show_confidence(Linear_list, linear_confidence)

    print('*' * 45 + '  Analysis circle ' + '*' * 45)
    show_confidence(circle_list, circle_confidence)

    print('*' * 45 + '  Analysis A ' + '*' * 45)
    show_confidence(A_list, A_confidence, 100, A_ID_2_STANDID_DICT)

    print('*' * 45 + '  Analysis B ' + '*' * 45)
    show_confidence(B_list, B_confidence, 100, B_ID_2_STANDID_DICT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Classification', type=str, default="./result/classification_results.csv", help='results of classification model')
    parser.add_argument('--Regression', type=str, default="./result/regression_results.csv", help='results of regression model')
    parser.add_argument('--save', type=str, default="./result", help='path to save results')
    parser.add_argument('--cutoff', type=int, default=5000, help='cutoff for regression model')
    args = parser.parse_args()

    main(args)