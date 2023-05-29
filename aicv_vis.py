# !/usr/bin/env python3
# -- coding: utf-8 --
""" Point cloud data visualization function

Author: fangchenyu
Date: 2022/12/27
"""
import os
import copy
from pathlib import Path
import json
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation

import pypcd
from utils.kitti_util import lidar_to_top, draw_top_image, show_lidar_topview_with_boxes



def get_scan_from_pcloud(pcloud):
    ''' 从pypcd的数据结构中获取点云数据，生成numpy.array格式
    pcloud: pypcd.PointCloud
    '''
    scan = np.empty((pcloud.points, 4), dtype=np.float32)
    scan[:, 0] = pcloud.pc_data['x']
    scan[:, 1] = pcloud.pc_data['y']
    scan[:, 2] = pcloud.pc_data['z']
    try:
        scan[:, 3] = pcloud.pc_data['intensity']
    except ValueError:
        scan[:, 3] = 255.0
    return scan


def load_pcd(f_pcd):
    ''' 从pcd文件读取点云数据，将数据到数据结构pypcd.PointCloud
    f_pcd: strin, pcd文件路径
    '''
    try:
        if isinstance(f_pcd, str) or isinstance(f_pcd, Path):
            pcloud = pypcd.PointCloud.from_path(f_pcd)
        else:
            raise TypeError(f'load_pcd do not support type {type(f_pcd)}')

    except AssertionError:
        print ("Assertion when load pcd: %s" % f_pcd)
        return None
    scan = get_scan_from_pcloud(pcloud)
    scan[:, 3] /= 255.0
    return scan


def read_label(anno_infos, sample_idx):
    info = copy.deepcopy(anno_infos.iloc[sample_idx, :])
    sample_idx = info['_id']
    get_item_list = ['points']
    result = json.loads(info['result'])
    num_obj = result['labelData']['markData']['numberCube3d']
    annos = result['labelData']['markData']['cube3d']
    loc, dims, rots, gt_names, trackId = [], [], [], [], []
    for i in range(num_obj):
        loc.append([annos[i]['position']['x'], annos[i]['position']['y'], annos[i]['position']['z']])
        dims.append(annos[i]['size'])
        rots.append(annos[i]['rotation']['phi'])
        gt_names.append(annos[i]['type'])
        trackId.append(annos[i]['trackId'])
    loc, dims, rots, gt_names, trackId = np.array(loc), np.array(dims), np.array(rots), np.array(gt_names), np.array(trackId)
    gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32) # [x, y, z, l, w, h, r_z]
    # gt_boxes_camera = box_utils.boxes3d_aicv_to_kitti_camera(gt_boxes_lidar) # [x, y, z, l, h, w, r]
    # annos_camera_coordinate = box_utils.boxes3d_kitti_camera_to_annos(gt_boxes_camera, gt_names, gt_boxes_lidar)
    
    info_dict = {
        'frame_id': sample_idx,
        'trackId': trackId,
        'gt_names': gt_names,
        'gt_boxes': gt_boxes_lidar,
        # 'annos': annos_camera_coordinate,
        # 'pcd_path': pcd_path
    }

    return info_dict


def vis_label(points, result, save_path):
    pred_names = result['gt_names']
    pred_bbox3d = result['gt_boxes']
    pred_tracks = result['trackId']
    pred_obj = {'names': pred_names, 'bbox3d': pred_bbox3d, 'trackID': pred_tracks}
    show_lidar_topview_with_boxes(points, pred_obj, vis_output_file=save_path)


def quaternion_to_rotation_matrix(quaternion):
    # 创建旋转对象
    rotation = Rotation.from_quat(quaternion)
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()

    return rotation_matrix


def trans_cam_to_pixel(cam_points, calib_cam_K):
    """ 将相机坐标系下的点转化为像素坐标系
    camera_points: shape(N, 3), 相机坐标系下的点
    calib_cam_K: list(9), 相机内参矩阵， [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    points_pixel: shape(N, 2) 像素坐标系下的点
    """
    calib_cam_K = np.array(calib_cam_K).reshape((3, 3))
    # 将相机坐标系下的点转换为像素坐标系
    points_pixel = np.dot(calib_cam_K, cam_points.T).T

    # 将齐次坐标转换为二维坐标
    points_pixel[:, 0] /= points_pixel[:, 2]
    points_pixel[:, 1] /= points_pixel[:, 2]
    points_pixel = points_pixel[:, :2]

    return points_pixel


def trans_lidar_to_cam(lidar_points, calib_cam2lidar, calib_cam_K):
    """ 将lidar坐标系下的点转化为相机坐标系下的点
    camera_points: np.array, shape(N, 3), 雷达坐标系下的点
    calib_cam2lidar: list(7), [Tx, Ty, Tz, Rx, Ry, Rz, Rw] 相机坐标系的pose
    calib_cam_K: list(9), 相机内参矩阵， [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    """
    # 解析转换信息
    Tx, Ty, Tz, Rx, Ry, Rz, Rw = calib_cam2lidar

    # 构建旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix([Rx, Ry, Rz, Rw])
    rotation_matrix_inv = np.linalg.inv(rotation_matrix) # lidar point to cam

    # 构建平移向量
    translation_vector = np.array([Tx, Ty, Tz]) # cam point to lidar
    translation_vector_inv = - np.array([Tx, Ty, Tz]) # lidar point to cam

    # 将LiDAR坐标系转化为相机坐标系下的点
    cam_points = np.dot(rotation_matrix_inv, lidar_points.T) + translation_vector_inv[:, np.newaxis]
    cam_points = cam_points.T # (3, N) -> (N, 3)

    # 将相机坐标系下的点转化为像素坐标系下的点
    pixel_points = trans_cam_to_pixel(cam_points, calib_cam_K)

    return pixel_points



def compute_box_3d(obj):
    # 3d bounding box dimensions
    l = obj[3]
    w = obj[4]
    h = obj[5]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    # corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj[0]
    corners_3d[1, :] = corners_3d[1, :] + obj[1]
    corners_3d[2, :] = corners_3d[2, :] + obj[2]
    # print('cornsers_3d: ', corners_3d)

    cam2lidar = [0.489518130031956, -0.02945436564130331, -0.4929393635788265, -0.500677789650434, 0.49376240367541463, -0.49772616641790407, 0.5077293599255113]
    K = np.array([3157.740152389458, 0, 1877.8239571322447, 0, 3157.74015238945, 1032.5844237793194, 0, 0, 1]).reshape((3, 3))

    corners_2d = trans_lidar_to_cam(corners_3d.T, cam2lidar, K)
    
    return corners_2d


# def show_image_with_boxes(img, objects, calib, show3d=True, depth=None, save_dir='output'):
# def show_image_with_boxes(img, objects, show3d=True, depth=None, save_dir='output'):
def show_image_with_boxes(img, objects, show3d=True, depth=None, save_dir=None):
    """ 对图像中的物体进行2d和3d框的可视化
        cv2: 默认色彩通道顺序为BGR
        PIL: 默认色彩通道顺序为RGB
        进行可视化展示时需要注意其色彩通道，必要时对其进行转化
    """
    import utils.kitti_util as utils

    img2 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    gt_names = objects['gt_names']
    objects = objects['gt_boxes']
    for i, obj in enumerate(objects):
        box3d_pts_2d = compute_box_3d(obj)

        # box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if gt_names[i] == "smallMot":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif gt_names[i] == "bigMot":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        # elif obj.type == "Cyclist":
        #     img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))
    
    if save_dir is not None: # 将可视化结果保存为图像
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, 'aicv-img-label.png'), img2)
    else: # 将可视化结果直接可视化
        show3d = True
        if show3d:
            cv2.imshow("3dbox", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img2


if __name__ == '__main__':
    # 1. 可视化lidar
    pcd_file = 'data/aicv-sample-ori/velodyne/at128_fusion.pcd'
    pc_data = load_pcd(pcd_file)
    # print(pc_data.shape)
    # img_bev = save_bev_image_by_point_cloud(pc_data, 'tmp.png')

    # 2. label读取
    info_path = 'data/aicv-sample-ori/label_02/result.txt'
    anno_infos = pd.read_csv(info_path, sep='\t')
    # print(anno_infos.shape)
    anno = read_label(anno_infos, sample_idx=0)
    # print(anno)

    # 3. label可视化
    vis_label(pc_data, anno, save_path='output/aicv-bev-label.png')

    # 4. 图像可视化
    img = cv2.imread('data/aicv-sample-ori/image_02/image.jpg')
    show_image_with_boxes(img, anno, save_dir='output')





    



