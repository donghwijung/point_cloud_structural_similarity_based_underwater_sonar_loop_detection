import numpy as np
import open3d as o3d

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