import argparse
import os
import copy
import pickle

import multiprocessing as multiproc

import numpy as np
from tqdm import tqdm

from auvlib.data_tools import std_data

from utils import read_bin
from proc_pc import accum_pcd_points_and_save_cropped_pcds_antarctica

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments parser.")

    parser.add_argument('--data_path', type=str, default="data", help='The path of data directory')
    parser.add_argument('--divide_size', type=int, default=15, help='The number of processes to execute in parallel')
    parser.add_argument('--points_size', type=int, default=8192, help='The number of points in a point cloud')
    args = parser.parse_args()

    divide_size = args.divide_size
    points_size = args.points_size

    dataset_name = "antarctica"
    print(f"{dataset_name} is selected")

    TRAIN_PINGS = [36000, 74180]
    TEST_PINGS = [74180, 77900]
    test_idx = [TRAIN_PINGS[0], TEST_PINGS[1]]
        
    exclude_recent_ids = 1000
    same_node_range = 50
    radius = 100.0

    seq_size = 500
    half_seq_size = seq_size // 2

    size = 135*2

    Id4 = np.eye(4)

    data_path = os.path.join(args.data_path, dataset_name)
    processed_dir_path = os.path.join(data_path, "processed")
    if not os.path.isdir(processed_dir_path):
        os.makedirs(processed_dir_path)

    std_cereal = std_data.mbes_ping.read_data(os.path.join(data_path, "antarctica_2019.cereal"))[test_idx[0]:test_idx[1]]

    centers_sample = np.empty((0,3))
    for i in range(len(std_cereal)):
        current_data = std_cereal[i]
        current_position = np.array(current_data.pos_)
        centers_sample = np.r_[centers_sample, current_position.reshape(1,3)]
    train_traj = read_bin(os.path.join(data_path, "train_trajectory.bin"))
    test_traj = read_bin(os.path.join(data_path, "test_trajectory.bin"))
    whole_traj = np.r_[train_traj, test_traj]
    positions = whole_traj

    loop_candidates_wrapper = np.empty((0,2), dtype=int)
    half_radius = 0.5*radius
    for i in tqdm(range(half_seq_size,positions.shape[0]-half_seq_size)):
        if i > exclude_recent_ids:
            src_position = positions[i]
            tgt_positions = positions[half_seq_size:i-exclude_recent_ids]
            dists = np.linalg.norm(tgt_positions-src_position,axis=1)
            loop_candidates = np.where((dists>0)&(dists<half_radius))[0]
            added_loop_candidates = np.zeros((loop_candidates.shape[0],2),dtype=int)
            added_loop_candidates[:,0] = i
            added_loop_candidates[:,1] = loop_candidates + half_seq_size
            loop_candidates_wrapper = np.r_[loop_candidates_wrapper, added_loop_candidates]

    selected_ids = []
    positive_pairs = []

    candidate_ids = np.random.choice(np.arange(loop_candidates_wrapper.shape[0]), size*10)
    for ci in tqdm(candidate_ids):
        src_id, tgt_id = loop_candidates_wrapper[ci]
        if (src_id not in selected_ids and tgt_id not in selected_ids):
            selected_ids.append(src_id)
            selected_ids.append(tgt_id)
            positive_pairs.append([src_id, tgt_id])
            if len(selected_ids) >= size:
                break

    positive_pairs = np.array(positive_pairs)
    negative_pairs = np.zeros((positive_pairs.shape[0],2), dtype=int)

    src_ids = copy.deepcopy(positive_pairs[:,0])
    tgt_ids = copy.deepcopy(positive_pairs[:,1])

    np.random.shuffle(src_ids)
    for i, src_id in tqdm(enumerate(src_ids)):
        np.random.shuffle(tgt_ids)
        for tgt_id in tgt_ids:
            src_odom_id = src_id
            tgt_odom_id = tgt_id
            src_position = positions[src_odom_id]
            tgt_position = positions[tgt_odom_id]
            dist = np.linalg.norm(src_position - tgt_position)
            if dist > radius * 2 and np.abs(src_id - tgt_id) > exclude_recent_ids:
                if src_id > tgt_id:
                    negative_pairs[i] = np.array([src_id, tgt_id])
                    break
    with open(os.path.join(processed_dir_path, "positive_pairs.pkl"), "wb") as f:
        pickle.dump(positive_pairs, f)
    with open(os.path.join(processed_dir_path, "negative_pairs.pkl"), "wb") as f:
        pickle.dump(negative_pairs, f)

    src_ids = copy.deepcopy(positive_pairs[:,0])
    tgt_ids = copy.deepcopy(positive_pairs[:,1])

    ids = src_ids
    suffix = "train"

    quotient_w_seq_size = ids.shape[0] // divide_size

    start_and_end_idx_w_seq_size = []
    for i in range(divide_size):
        start_id = quotient_w_seq_size * i
        if i < divide_size - 1:
            end_id = quotient_w_seq_size * (i+1)
        else:
            end_id = ids.shape[0]
        start_and_end_idx_w_seq_size.append([start_id, end_id])

    th_list = []
    for i in range(divide_size):
        
        th = multiproc.Process(target=accum_pcd_points_and_save_cropped_pcds_antarctica, args=(std_cereal, ids[start_and_end_idx_w_seq_size[i][0]:start_and_end_idx_w_seq_size[i][1]],\
                                                                                    half_seq_size, points_size, processed_dir_path, radius, True, suffix))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    ids = tgt_ids
    suffix = "test"

    th_list = []
    for i in range(divide_size):
        
        th = multiproc.Process(target=accum_pcd_points_and_save_cropped_pcds_antarctica, args=(std_cereal, ids[start_and_end_idx_w_seq_size[i][0]:start_and_end_idx_w_seq_size[i][1]],\
                                                                                    half_seq_size, points_size, processed_dir_path, radius, True, suffix))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    print("Done")