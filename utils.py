import numpy as np
import open3d as o3d

def process_timestamps(timestamps, divider=1.0):
    processed_timestamps = timestamps / divider
    processed_timestamps -= processed_timestamps[0]
    time_diffs = np.zeros(processed_timestamps.shape)
    time_diffs[1:] = processed_timestamps[1:] - processed_timestamps[:-1]
    return processed_timestamps, time_diffs

def find_timestamp_idx(source_t, target_ts):
    time_diffs = np.abs(target_ts - source_t)
    return time_diffs.argmin()

def pcd_points_to_pcd(pcd_points, c=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    if c is not None:
        pcd.paint_uniform_color(c)
    return pcd

def read_bin(file_name, dtype=None):
    if not dtype:
        dtype=np.float32
    pcd_np = np.fromfile(file_name, dtype=dtype).reshape((-1, 3))
    return pcd_np