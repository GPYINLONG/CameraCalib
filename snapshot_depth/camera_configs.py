# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年04月25日
"""
import cv2
import numpy as np

left_camera_matrix = np.array(
    [[532.2904, .0000, 322.8069],
     [.0000, 531.1643, 225.8576],
     [.0000, .0000, 1.0000]])  # 内参矩阵（matlab数据需要进行转置）

left_camera_distortion = np.array([-.0040, .0703, -.0047, -.0017, .0000])  # 畸变矩阵k1,k2,p1,p2,k3

right_camera_matrix = np.array(
    [[530.4551, .0000, 352.2014],
     [.0000, 529.8504, 233.7585],
     [.0000, .0000, 1.0000]])

right_camera_distortion = np.array([-.0043, .0191, .0004, -.0028, .0000])  # [-.0670, .4833, .0003, -.0031, -.9868]

R = np.array(
    [[1.0000, -.0013, .0039],
     [.0013, 1.0000, -.0051],
     [-.0039, .0051, 1.0000]])  # 相机2相对于相机1的旋转矩阵（matlab数据需要进行转置）

T = np.array([-121.5017, -.1081, -.1592])  # 相机2相对于相机1的偏移矩阵

size = (640, 480)  # 图像尺寸

# 进行立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_camera_matrix, left_camera_distortion, right_camera_matrix, right_camera_distortion,
    size, R, T
)

# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_camera_distortion, R1, P1, size, cv2.CV_32FC1
)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_camera_distortion, R2, P2, size, cv2.CV_32FC1
)
