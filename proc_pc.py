import os

import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from utils import pcd_points_to_pcd, find_timestamp_idx
from pointssim import pcd_fuse_points

def build_pcd_target_idx_dict(sub_pcd_file_list, start_id, initial_pcd_timestamp, target_timestamps, result):
    sub_pcd_target_idx_dict = {}
    for pf_i, pcd_file in tqdm(enumerate(sub_pcd_file_list)): 
        pcd_timestamp = float(pcd_file.replace(".pcd", ""))
        modi_pf_i = start_id + pf_i 
        pcd_timestamp -= initial_pcd_timestamp
        target_id = find_timestamp_idx(pcd_timestamp, target_timestamps)
        sub_pcd_target_idx_dict[modi_pf_i] = target_id
    result.append(sub_pcd_target_idx_dict)

def square_crop(pcd_points, center, radius):
    center_xy = center[:2]
    pcd_points_xy = pcd_points[:,:2]
    cropped_points = pcd_points[(pcd_points_xy[:,0]>center_xy[0]-radius)&(pcd_points_xy[:,0]<center_xy[0]+radius)&(pcd_points_xy[:,1]>center_xy[1]-radius)&(pcd_points_xy[:,1]<center_xy[1]+radius)]
    return cropped_points

def pcd_id_to_pcd_points(pcd_id, std_cereal):
    current_data = std_cereal[pcd_id]
    pcd_points = np.array(current_data.beams)
    return pcd_points

def pcd_id_to_transformed_pcd_points(pcd_id, pcd_file, pcd_dir_path, poses, pcd_odom_idx_dict, criteria_pose_inv):
    pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir_path, pcd_file))
    pcd_points = np.asarray(pcd.points)
    pcd_pose = criteria_pose_inv @ poses[pcd_odom_idx_dict[pcd_id]]
    pcd_points_homo = np.c_[pcd_points, np.ones(pcd_points.shape[0])]
    transformed_pcd_points = (pcd_pose @ pcd_points_homo.T).T[:,:3]
    return transformed_pcd_points

def accum_pcd_points_and_save_cropped_pcds_seaward(pcd_files_list, pcd_ids, poses, pcd_odom_idx_dict, size, points_size, processed_dir_path, pcd_dir_path, radius, global_transform=True, file_prefix="train"):
    for curr_pcd_id in tqdm(pcd_ids):
        pcd_file_name = pcd_files_list[curr_pcd_id]
        accumed_pcd_points = np.empty((0,3))

        criteria_pose = poses[pcd_odom_idx_dict[curr_pcd_id]]
        criteria_pose_inv = np.linalg.inv(criteria_pose)
        
        ## Current
        transformed_pcd_points = pcd_id_to_transformed_pcd_points(curr_pcd_id, pcd_file_name, pcd_dir_path, poses, pcd_odom_idx_dict, criteria_pose_inv)
        criteria_pose_inv = np.linalg.inv(criteria_pose)
        accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]
        
        ## Next
        for i in range(1,size+1):
            global_pcd_id = curr_pcd_id + i
            pcd_file_name = pcd_files_list[global_pcd_id]
            transformed_pcd_points = pcd_id_to_transformed_pcd_points(global_pcd_id, pcd_file_name, pcd_dir_path, poses, pcd_odom_idx_dict, criteria_pose_inv)
            accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]
        
        ## Previous
        for i in range(1,size+1):
            global_pcd_id = curr_pcd_id - i
            pcd_file_name = pcd_files_list[global_pcd_id]
            transformed_pcd_points = pcd_id_to_transformed_pcd_points(global_pcd_id, pcd_file_name, pcd_dir_path, poses, pcd_odom_idx_dict, criteria_pose_inv)
            accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]

        unique_accumed_pcd_points = pcd_fuse_points(accumed_pcd_points)
        cropped_points = square_crop(unique_accumed_pcd_points, np.zeros(3), radius)

        if global_transform:
            cropped_points_homo = np.c_[cropped_points, np.ones(cropped_points.shape[0])]
            cropped_points = (criteria_pose @ cropped_points_homo.T).T[:,:3]
            center = criteria_pose[:3,3]
        else:
            center = np.zeros(3)

        center = center.reshape(1,3)

        sampled_points = cropped_points[np.random.choice(cropped_points.shape[0], points_size, replace=False)]
        sampled_points_w_center = np.r_[center, sampled_points]
        processed_saved_pcd = pcd_points_to_pcd(sampled_points_w_center)
        o3d.io.write_point_cloud(os.path.join(processed_dir_path, f"{file_prefix}_{curr_pcd_id}.pcd"), processed_saved_pcd)

def accum_pcd_points_and_save_cropped_pcds_antarctica(std_cereal, pcd_ids, size, points_size, processed_dir_path, radius, global_transform=True, file_prefix="train"):
    for curr_pcd_id in tqdm(pcd_ids):
        accumed_pcd_points = np.empty((0,3))

        criteria_data = std_cereal[curr_pcd_id]
        criteria_pose = np.eye(4)
        criteria_euler_angles = np.array([criteria_data.roll_, criteria_data.pitch_, criteria_data.heading_])
        criteria_position = np.array(criteria_data.pos_)
        criteria_pose[:3,3] = criteria_position
        criteria_pose[:3,:3] = Rotation.from_euler("xyz", criteria_euler_angles).as_matrix()
        
        ## Current
        transformed_pcd_points = pcd_id_to_pcd_points(curr_pcd_id, std_cereal)
        accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]
        
        ## Next
        for i in range(1,size+1):
            global_pcd_id = curr_pcd_id + i
            transformed_pcd_points = pcd_id_to_pcd_points(global_pcd_id, std_cereal)
            accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]
        
        ## Previous
        for i in range(1,size+1):
            global_pcd_id = curr_pcd_id - i
            transformed_pcd_points = pcd_id_to_pcd_points(global_pcd_id, std_cereal)
            accumed_pcd_points = np.r_[accumed_pcd_points, transformed_pcd_points]

        unique_accumed_pcd_points = pcd_fuse_points(accumed_pcd_points)

        if global_transform:
            center = criteria_pose[:3,3]
        else:
            center = np.zeros(3)

        cropped_points = square_crop(unique_accumed_pcd_points, center, radius)
        center = center.reshape(1,3)
        try:
            sampled_points = cropped_points[np.random.choice(cropped_points.shape[0], points_size, replace=False)]
        except ValueError:
            print(cropped_points.shape)
            sampled_points = np.empty((0,3))
        sampled_points_w_center = np.r_[center, sampled_points]
        processed_saved_pcd = pcd_points_to_pcd(sampled_points_w_center)
        o3d.io.write_point_cloud(os.path.join(processed_dir_path, f"{file_prefix}_{curr_pcd_id}.pcd"), processed_saved_pcd)