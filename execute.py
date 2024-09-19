import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from pointssim import pairs_to_sims_wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments parser.")

    parser.add_argument('--data_path', type=str, default="data", help='The path of data directory')
    parser.add_argument('--data_id', type=int, default=0, help='The index of dataset')
    parser.add_argument('--neighborhood_size', type=int, default=100, help='The number of nearest neighborhoods')
    parser.add_argument('--score_threshold', type=float, default=2.95, help='The threshold to determine whether a loop or not')
    args = parser.parse_args()

    dataset_names = ["antarctica","wiggles_bank","north_river","dutch_harbor","beach_pond"] ## wiggles_bank, north_river, dutch_harbor, beach_pond are seaward dataset
    dataset_name = dataset_names[args.data_id]

    estimator_types = ["VAR", "Mean"]
    pooling_types = ["Mean"]
    neighborhood_size = args.neighborhood_size

    if "antarctica" in dataset_name:
        radius = 100.0
    else:
        radius = 10.0

    score_threshold = args.score_threshold
    plot_pr_curve = True

    if "/" in dataset_name:
        dataset_name = dataset_name.split("/")[0]

    data_path = os.path.join(args.data_path, dataset_name)
    processed_dir_path = os.path.join(data_path, "processed")

    print(f"Test on {dataset_name} dataset. {data_path} {neighborhood_size} {score_threshold}")   

    with open(os.path.join(processed_dir_path, "positive_pairs.pkl"), "rb") as f:
        test_pair_positive = pickle.load(f)
    with open(os.path.join(processed_dir_path, "negative_pairs.pkl"), "rb") as f:
        test_pair_negative = pickle.load(f)

    test_pair_positive = np.array(test_pair_positive)
    test_pair_negative = np.array(test_pair_negative)

    print("Proceed with positive loop pairs")
    negative_sims_wrapper = pairs_to_sims_wrapper(test_pair_negative, processed_dir_path, neighborhood_size, radius, estimator_types, pooling_types)
    print("Proceed with negative loop pairs")
    positive_sims_wrapper = pairs_to_sims_wrapper(test_pair_positive, processed_dir_path, neighborhood_size, radius, estimator_types, pooling_types)

    sum_positive_sims_wrapper = np.sum(positive_sims_wrapper.reshape(-1,3*2), axis=1)
    sum_negative_sims_wrapper = np.sum(negative_sims_wrapper.reshape(-1,3*2), axis=1)

    tp_list = test_pair_positive[np.where(sum_positive_sims_wrapper >= score_threshold)]
    fn_list = test_pair_positive[np.where(sum_positive_sims_wrapper < score_threshold)]
    tp_probs = sum_positive_sims_wrapper[np.where(sum_positive_sims_wrapper >= score_threshold)]
    fn_probs = sum_positive_sims_wrapper[np.where(sum_positive_sims_wrapper < score_threshold)]

    fp_list = test_pair_negative[np.where(sum_negative_sims_wrapper >= score_threshold)]
    tn_list = test_pair_negative[np.where(sum_negative_sims_wrapper < score_threshold)]
    fp_probs = sum_negative_sims_wrapper[np.where(sum_negative_sims_wrapper >= score_threshold)]
    tn_probs = sum_negative_sims_wrapper[np.where(sum_negative_sims_wrapper < score_threshold)]

    print(f"TP: {len(tp_list)}, FP: {len(fp_list)}, TN: {len(tn_list)}, FN: {len(fn_list)}")

    if plot_pr_curve:
        y_true = np.append(np.ones(len(tp_list) + len(fn_list),dtype=int),np.zeros(len(fp_list) + len(tn_list),dtype=int))
        y_pred = np.append(np.append(tp_probs,fn_probs), np.append(tn_probs,fp_probs))
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        line_width = 2
        plt.rcParams.update({'font.size': 14})

        plt.plot(recall, precision, label='Our method', linewidth=line_width, c="tab:blue")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.grid(False)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(f"{dataset_name} PR curve")
        plt.show()