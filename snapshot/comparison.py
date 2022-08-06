# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年05月17日
"""

import cv2
import numpy as np
from snapshot_depth import camera_configs
import math


def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _3D = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        print('世界坐标xyz是：', _3D[y][x][0] / 1000.0, _3D[y][x][1] / 1000.0, _3D[y][x][2] / 1000.0, 'm')

        distance = math.sqrt(_3D[y][x][0] ** 2 + _3D[y][x][1] ** 2 + _3D[y][x][2] ** 2) / 1000.0
        print('距离是：', distance, 'm')


def update(val=0):
    blocksize = cv2.getTrackbarPos('blockSize', 'img')
    num = cv2.getTrackbarPos('Numdisparities(x16)', 'img') * 16
    if blocksize % 2 == 0:
        blocksize += 1
    if blocksize < 5:
        blocksize = 5
    if num < 16:
        num = 16
    uniquenessratio = cv2.getTrackbarPos('uniquenessRatio', 'img')
    specklewindowsize = cv2.getTrackbarPos('speckleWindowSize', 'img')
    specklerange = cv2.getTrackbarPos('speckleRange', 'img')
    maxdiff = cv2.getTrackbarPos('MaxDiff', 'img')

    stereo_SGBM.setBlockSize(blocksize)
    stereo_SGBM.setNumDisparities(num)
    stereo_BM.setNumDisparities(num)
    stereo_BM.setBlockSize(blocksize)
    stereo_SGBM.setUniquenessRatio(uniquenessratio)
    stereo_BM.setUniquenessRatio(uniquenessratio)
    stereo_BM.setSpeckleWindowSize(specklewindowsize)
    stereo_SGBM.setSpeckleWindowSize(specklewindowsize)
    stereo_BM.setSpeckleRange(specklerange)
    stereo_SGBM.setSpeckleRange(specklerange)
    stereo_BM.setDisp12MaxDiff(maxdiff)
    stereo_SGBM.setDisp12MaxDiff(maxdiff)

    dp_bm = stereo_BM.compute(imgL, imgR)
    dp_sgbm = stereo_SGBM.compute(imgL, imgR)
    disp_bm = cv2.normalize(dp_bm, dp_bm, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化函数算法
    disp_sgbm = cv2.normalize(dp_sgbm, dp_sgbm, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threeD_BM = cv2.reprojectImageTo3D(dp_bm, camera_configs.Q,
                                       handleMissingValues=True) * 16
    threeD_SGBM = cv2.reprojectImageTo3D(dp_sgbm, camera_configs.Q,
                                         handleMissingValues=True) * 16
    cv2.setMouseCallback('depth_BM', callback, threeD_BM)  # 第一个参数，表示将要操作的面板名，第二个参数是回调函数名，第三个是给回调函数的参数
    cv2.setMouseCallback('depth_SGBM', callback, threeD_SGBM)

    cv2.imshow('depth_BM', disp_bm)
    cv2.imshow('depth_SGBM', disp_sgbm)
    cv2.imshow('img', img)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
    elif cv2.waitKey(0) & 0xff == ord('s'):
        if cv2.imwrite('shot/depth_BM.jpg', disp_bm) \
           and cv2.imwrite('shot/depth_SGBM.jpg', disp_sgbm):
            print('SnapShot saved.')


if __name__ == '__main__':
    imgL = cv2.imread(r'shot/rectifiedLeft.jpg', 0)
    imgR = cv2.imread(r'shot/rectifiedRight.jpg', 0)
    img = np.hstack((cv2.imread(r'shot/frameL.jpg'), cv2.imread(r'shot/frameR.jpg')))
    cv2.namedWindow('depth_BM')
    cv2.namedWindow('depth_SGBM')
    cv2.namedWindow('img')
    cv2.moveWindow('depth_BM', 0, 0)
    cv2.moveWindow('depth_SGBM', 800, 0)
    cv2.resizeWindow('depth_BM', 640, 480)
    cv2.resizeWindow('depth_SGBM', 640, 480)

    cv2.createTrackbar('Numdisparities(x16)', 'img', 1, 15, update)
    cv2.createTrackbar('blockSize', 'img', 5, 49, update)
    cv2.createTrackbar('uniquenessRatio', 'img', 5, 15, update)  # 最佳计算成本函数值应该“赢”第二个最佳值的百分比，通常设置在5-15
    cv2.createTrackbar('speckleWindowSize', 'img', 50, 200, update)  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。0为禁用斑点过滤，否则设置在50-200的范围
    cv2.createTrackbar('speckleRange', 'img', 1, 16, update)
    cv2.createTrackbar('MaxDiff', 'img', 1, 200, update)  # 左右视差检查允许的最大差异，为负值时禁用检查
    stereo_BM = cv2.StereoBM_create(
        numDisparities=16,
        blockSize=5,
    )

    stereo_SGBM = cv2.StereoSGBM_create(
        blockSize=5,
        numDisparities=16,
        P1=600,
        P2=2400,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    stereo_BM.setMinDisparity(0)
    stereo_SGBM.setMinDisparity(0)
    stereo_BM.setROI1(camera_configs.validPixROI1)
    stereo_BM.setROI2(camera_configs.validPixROI2)

    update()
