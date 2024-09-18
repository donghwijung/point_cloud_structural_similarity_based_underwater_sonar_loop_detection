import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from utils import pcd_points_to_pcd

def pcd_fuse_points(pc_in):
    pc_out = np.unique(pc_in, axis=0)
    return pc_out

def feature_map(quant,estimator_types=["VAR", "Mean"]):
    f_map = np.zeros((quant.shape[0], len(estimator_types)))
    for et_i, e_type in enumerate(estimator_types):
        if e_type == "Mean":
            f_map[:, et_i] = quant.mean(axis=1)
        elif e_type == "VAR":
            f_map[:, et_i]= quant.var(axis=1,ddof=1)
    return f_map

def error_map(f_map_y, f_map_x, id_yx, const):
    numerator = abs(f_map_x[id_yx] - f_map_y)
    denominator = (np.max([np.abs(f_map_x[id_yx]), np.abs(f_map_y)], axis=0)) + const
    return numerator / denominator

def pooling(q_map, pooling_types):
    score = np.zeros((1, len(pooling_types)))
    for pt_i, p_type in enumerate(pooling_types):
        if p_type == "Mean":
            score[:, pt_i] = q_map.mean()
    return score

def ssim_score(feat_map_a, feat_map_b, id_ba, id_ab, estimator_types, pooling_types, const=2.2204e-16):
    ssim_ba = np.zeros((len(estimator_types), len(pooling_types)))
    for et_i, _ in enumerate(estimator_types):
        error_map_ba =  error_map(feat_map_b[:,et_i], feat_map_a[:,et_i], id_ba, const)
        ssim_map_ba = 1 - error_map_ba
        ssim_ba[et_i,:] = pooling(ssim_map_ba, pooling_types)
    
    ssim_ab = np.zeros((len(estimator_types), len(pooling_types)))
    for et_i, _ in enumerate(estimator_types):
        error_map_ab =  error_map(feat_map_a[:,et_i], feat_map_b[:,et_i], id_ab, const)
        ssim_map_ab = 1 - error_map_ab
        ssim_ab[et_i,:] = pooling(ssim_map_ab, pooling_types)

    ssim_sym = np.minimum(ssim_ba, ssim_ab)
    return ssim_sym

def cal_norm_quant(pcd_normals, ids, neighborhood_size):
    cal_one = np.multiply(pcd_normals[ids.transpose(),:].reshape(-1,3),np.tile(pcd_normals, (neighborhood_size, 1)))
    cal_two = np.sqrt(np.sum(pcd_normals[ids.transpose(),:].reshape(-1,3)**2,axis=1))
    cal_three = np.sqrt(np.sum(np.tile(pcd_normals,(neighborhood_size, 1))**2,axis=1))

    ns = 1 - 2*np.arccos(np.abs(np.divide(np.sum(cal_one,axis=1),np.multiply(cal_two,cal_three))))/np.pi
    ns = np.nan_to_num(ns, nan=1)
    norm_quant = ns.reshape((int(ns.shape[0]/neighborhood_size), neighborhood_size), order="F")
    norm_quant = norm_quant[:,1:]
    return norm_quant, ns

def poly_func(xy, a,b,c,d,e,f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y

def calculate_surface_curvature(pcd, radius, max_nn):
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    covs = np.asarray(pcd.covariances)
    vals, vecs = np.linalg.eig(covs)
    curvature = np.min(vals, axis=1)/np.sum(vals, axis=1)
    return curvature

def pc_estimate_norm_curv_qfit_w_o3d(pc_in, radius, max_nn):
    curvs = calculate_surface_curvature(pc_in, radius, max_nn)
    pc_in.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(pc_in.normals)
    return normals, curvs

def pc_estimate_norm_curv_qfit(pc_in_points):
    search_size = np.round(0.1 * np.max(np.max(pc_in_points, axis=0) - np.min(pc_in_points, axis=0), axis=0))
    search_size *= 5 ## Different from original (search scale up)
    geom = pc_in_points
    pcd_in_tree = cKDTree(pc_in_points)

    range_search_ids = pcd_in_tree.query_ball_point(pc_in_points, r=search_size)

    normals = np.zeros((geom.shape[0], 3))
    curvatures = np.zeros((geom.shape[0], 1))

    for i in range(geom.shape[0]):
        point = geom[i]
        point_neighb = geom[range_search_ids[i], :]

        covariance_matrix = np.cov(point_neighb.transpose(), ddof=0)
        if np.sum(np.isnan(covariance_matrix)) > 1:
            return
        _, eigvecs = np.linalg.eig(covariance_matrix)
        if eigvecs.shape[1] != 3:
            return
        data_transf = (point_neighb - np.mean(point_neighb)) @ eigvecs
        point_transf = (point - np.mean(point_neighb)) @ eigvecs

        xyz = data_transf - point_transf
        p,_ = curve_fit(poly_func, (xyz[:,0], xyz[:,1]), xyz[:,2])
        if all(np.isnan(p)):
            return

        p20 = p[3]
        p11 = p[5]
        p10 = p[1]
        p02 = p[4]
        p01 = p[2]

        grad_x = p10
        grad_y = p01

        normal = [-grad_x, -grad_y, 1]
        normal = np.divide(normal, np.linalg.norm(normal))

        normal = normal.reshape(1,3) @ eigvecs.transpose()

        normals[i,:] = np.divide(normal, np.linalg.norm(normal))
        curvatures[i]= ((1+p10**2)*p20+(1+p01**2)*p02-4*p20*p02*p11) / (1+p01**2+p10**2)**(3/2)
    return normals, curvatures

def pointssim(pcd_a_tree, pcd_b_tree, pcd_points_a, pcd_points_b, geom_feat_map_a, geom_feat_map_b, norm_feat_map_a, norm_feat_map_b, curv_feat_map_a, curv_feat_map_b, estimator_types, pooling_types):
    _, id_ba = pcd_a_tree.query(pcd_points_b, k=1)
    _, id_ab = pcd_b_tree.query(pcd_points_a, k=1)

    geom_sym = ssim_score(geom_feat_map_a, geom_feat_map_b, id_ba, id_ab, estimator_types, pooling_types)
    norm_sym = ssim_score(norm_feat_map_a, norm_feat_map_b, id_ba, id_ab, estimator_types, pooling_types)
    curv_sym = ssim_score(curv_feat_map_a, curv_feat_map_b, id_ba, id_ab, estimator_types, pooling_types)
    return geom_sym, norm_sym, curv_sym

def pcd_id_to_features(processed_dir_path, src_id, neighborhood_size, radius, estimator_types, prefix="train"):
    src_file_path = os.path.join(processed_dir_path, f"{prefix}_{src_id}.pcd")
    src_origin_pcd = o3d.io.read_point_cloud(src_file_path)
    src_points = np.asarray(src_origin_pcd.points[1:]) ## exclude the center
    src_points = pcd_fuse_points(src_points)
    src_pcd = pcd_points_to_pcd(src_points)
    src_pcd_tree = cKDTree(src_points)
    dist_ss, id_ss = src_pcd_tree.query(src_points, k=neighborhood_size)
    src_geom_quant = dist_ss[:,1:] ## except for the point itself
    src_geom_feat_map = feature_map(src_geom_quant, estimator_types)
    max_nn = neighborhood_size
    src_norm, src_curv = pc_estimate_norm_curv_qfit_w_o3d(src_pcd, radius, max_nn)
    src_norm_quant, _ = cal_norm_quant(src_norm, id_ss, neighborhood_size)
    src_curv_quant = src_curv[id_ss].reshape(-1,neighborhood_size)
    src_norm_feat_map = feature_map(src_norm_quant, estimator_types)
    src_curv_feat_map = feature_map(src_curv_quant, estimator_types)
    return (src_pcd_tree, src_points, src_geom_feat_map, src_norm_feat_map, src_curv_feat_map)

def pairs_to_sims_wrapper(pairs, processed_dir_path, neighborhood_size, radius, estimator_types, pooling_types):
    sims_wrapper = np.empty((0,3,2))
    for tpp in tqdm(pairs):
        src_id = tpp[0]
        tgt_id = tpp[1]
        src_feats = pcd_id_to_features(processed_dir_path, src_id, neighborhood_size, radius, estimator_types, "train")
        tgt_feats = pcd_id_to_features(processed_dir_path, tgt_id, neighborhood_size, radius, estimator_types, "test")
        geom_sym, norm_sym, curv_sym = pointssim(src_feats[0], tgt_feats[0],src_feats[1], tgt_feats[1],src_feats[2], tgt_feats[2],src_feats[3], tgt_feats[3],src_feats[4], tgt_feats[4], estimator_types, pooling_types)
        sym_wrapper = np.zeros((3,2))
        sym_wrapper[0] = geom_sym.reshape(1,2)
        sym_wrapper[1] = norm_sym.reshape(1,2)
        sym_wrapper[2] = curv_sym.reshape(1,2)
        sims_wrapper = np.r_[sims_wrapper, sym_wrapper.reshape(1,3,2)]
    return sims_wrapper