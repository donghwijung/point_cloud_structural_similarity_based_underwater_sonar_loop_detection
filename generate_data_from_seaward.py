import argparse
import os
import copy
import pickle

import multiprocessing as multiproc

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from utils import process_timestamps
from proc_pc import build_pcd_target_idx_dict, accum_pcd_points_and_save_cropped_pcds_seaward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments parser.")

    parser.add_argument('--data_path', type=str, default="data", help='The path of data directory')
    parser.add_argument('--data_idx', type=int, default=0, help='The index of dataset')
    parser.add_argument('--divide_size', type=int, default=15, help='The number of processes to execute in parallel')
    parser.add_argument('--points_size', type=int, default=8192, help='The number of points in a point cloud')
    args = parser.parse_args()

    divide_size = args.divide_size
    points_size = args.points_size

    dataset_names = ["wiggles_bank","north_river","dutch_harbor","beach_pond"]
    dataset_name = dataset_names[args.data_idx]
    print(f"{dataset_name} is selected")

    if dataset_name.startswith("wiggles_bank"):
        test_idx = [0,94352] ## wigges_bank (before the lost part of sonar data)
    elif dataset_name.startswith("dutch_harbor"):
        test_idx = [0,91404] ## dutch_harbor (before the lost part of sonar data)
    elif dataset_name.startswith("north_river"):
        test_idx = [0,145844]
    elif dataset_name.startswith("beach_pond"):
        test_idx = [0,123696]

    exclude_recent_ids = 1000
    same_node_range = 100
    radius = 10.0

    seq_size = 500
    half_seq_size = seq_size // 2

    size = 135*2

    Id4 = np.eye(4)

    data_path = os.path.join(args.data_path, dataset_name)
    pcd_dir_path = os.path.join(data_path, "pcds")
    processed_dir_path = os.path.join(data_path, "processed")
    if not os.path.isdir(processed_dir_path):
        os.makedirs(processed_dir_path)

    pcd_files_list = os.listdir(pcd_dir_path)
    pcd_file_ids_list = [float(pf.replace(".pcd", "")) for pf in pcd_files_list]

    pcd_file_ids_list = np.array(pcd_file_ids_list)
    pcd_files_list = np.array(pcd_files_list)
    pcd_files_list = pcd_files_list[np.argsort(pcd_file_ids_list)[test_idx[0]:test_idx[1]]]
    pcd_file_ids_list = pcd_file_ids_list[np.argsort(pcd_file_ids_list)[test_idx[0]:test_idx[1]]]

    pcd_timestamps = np.zeros(test_idx[1]-test_idx[0])

    for pf_i, pcd_file_id in tqdm(enumerate(pcd_file_ids_list)):
        pcd_timestamp = pcd_file_id
        if pf_i == 0:
            initial_pcd_timestamp = pcd_timestamp
        pcd_timestamp -= initial_pcd_timestamp
        pcd_timestamps[pf_i] = pcd_timestamp

    odom_data = np.loadtxt(os.path.join(data_path, "odom.csv"), usecols=[2,5,6,7,8,9,10,11,48,49,50,51,52,53], skiprows=1, delimiter=",") ## timestamp, pos, quat (x,y,z,w), lin_vels, ang_vels
    odom_timestamps = odom_data[:,0]
    odom_timestamps, odom_time_diffs = process_timestamps(odom_timestamps, 10**9)

    odom_filtered_idx = np.where((pcd_timestamps[0] <= odom_timestamps)&(odom_timestamps <= pcd_timestamps[-1]))
    odom_data = odom_data[odom_filtered_idx]
    odom_timestamps = odom_timestamps[odom_filtered_idx]
    odom_time_diffs = odom_time_diffs[odom_filtered_idx]

    positions = odom_data[:,1:4]
    orientations = odom_data[:,4:8]
    euler_angles = Rotation.from_quat(orientations).as_euler("xyz")
    velocities = odom_data[:,8:11]
    ang_vels = odom_data[:,11:]
    rot_mats = Rotation.from_quat(orientations).as_matrix()

    initial_rot_mat = rot_mats[0]
    initial_rot_mat_inv = np.linalg.inv(initial_rot_mat)
    rot_mats = initial_rot_mat_inv @ rot_mats
    positions = (initial_rot_mat_inv @ (positions - positions[0]).T).T

    poses = np.swapaxes(np.dstack([Id4] * positions.shape[0]), 0, 2)
    poses[:,:3,3] = positions
    poses[:,:3,:3] = rot_mats

    transforms = np.swapaxes(np.dstack([Id4] * positions.shape[0]), 0, 2)
    for p_i, pose in enumerate(poses):
        if p_i > 0:
            transforms[p_i] = np.linalg.inv(poses[p_i-1]) @ poses[p_i]

    test_pcd_ids = np.arange(len(pcd_files_list)).tolist()
    test_pcd_files_list = [pcd_files_list[i] if i in test_pcd_ids else None for i in range(len(pcd_files_list))]

    quotient = len(test_pcd_files_list) // divide_size

    start_and_end_idx = []
    for i in range(divide_size):
        start_id = quotient * i
        if i < divide_size - 1:
            end_id = quotient * (i+1)
        else:
            end_id = len(test_pcd_files_list)
        start_and_end_idx.append([start_id, end_id])

    target_timestamps = odom_timestamps

    manager = multiproc.Manager()
    result = manager.list()
    th_list = []
    for i in range(divide_size):
        th = multiproc.Process(target=build_pcd_target_idx_dict, args=(pcd_files_list[start_and_end_idx[i][0]:start_and_end_idx[i][1]], start_and_end_idx[i][0], initial_pcd_timestamp, target_timestamps, result))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    pcd_odom_idx_dict = {}
    for r in result:
        pcd_odom_idx_dict.update(r)

    pcd_positions = np.zeros((len(test_pcd_files_list), 3))
    for i in tqdm(range(len(test_pcd_files_list))):
        pcd_positions[i] = positions[pcd_odom_idx_dict[i]]

    loop_candidates_wrapper = np.empty((0,2), dtype=int)
    half_radius = 0.5*radius
    for i in tqdm(range(half_seq_size,len(test_pcd_files_list)-half_seq_size)):
        if i > exclude_recent_ids:
            src_position = pcd_positions[i]
            tgt_positions = pcd_positions[half_seq_size:i-exclude_recent_ids]
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
            src_odom_id = pcd_odom_idx_dict[src_id]
            tgt_odom_id = pcd_odom_idx_dict[tgt_id]
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
        
        th = multiproc.Process(target=accum_pcd_points_and_save_cropped_pcds_seaward, args=(test_pcd_files_list, ids[start_and_end_idx_w_seq_size[i][0]:start_and_end_idx_w_seq_size[i][1]],\
                                                                                    poses, pcd_odom_idx_dict,
                                                                                    half_seq_size, points_size, processed_dir_path, pcd_dir_path,\
                                                                                        radius, True, suffix))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    ids = tgt_ids
    suffix = "test"

    th_list = []
    for i in range(divide_size):
        
        th = multiproc.Process(target=accum_pcd_points_and_save_cropped_pcds_seaward, args=(test_pcd_files_list, ids[start_and_end_idx_w_seq_size[i][0]:start_and_end_idx_w_seq_size[i][1]],\
                                                                                    poses, pcd_odom_idx_dict,
                                                                                    half_seq_size, points_size, processed_dir_path, pcd_dir_path,\
                                                                                        radius, True, suffix))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    print("Done")