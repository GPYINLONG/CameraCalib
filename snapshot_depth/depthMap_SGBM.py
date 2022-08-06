# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年04月30日
"""

import camera_configs
import cv2
import math



cv2.namedWindow('left')
cv2.namedWindow('right')
cv2.namedWindow('depth')
cv2.moveWindow('left', 0, 0)
cv2.moveWindow('right', 800, 0)
cv2.createTrackbar('num(x16)', 'depth', 1, 15, lambda x: None)
cv2.createTrackbar('blockSize', 'depth', 5, 49, lambda x: None)
camera = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

# 设置分辨率，左右的摄像机同一频率，同一设备ID
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 鼠标回调函数，添加点击事件，打印当前点的距离
def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _3D = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        print('世界坐标xyz是：', _3D[y][x][0]/1000.0, _3D[y][x][1]/1000.0, _3D[y][x][2]/1000.0, 'm')

        distance = math.sqrt(_3D[y][x][0]**2 + _3D[y][x][1]**2 + _3D[y][x][2]**2)/1000.0
        print('距离是：', distance, 'm')


while True:
    ret, frame = camera.read()
    # 裁剪坐标
    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    if not ret:
        break

    imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # 根据更正对map图片进行重构
    img1_rectified = cv2.remap(imgL, camera_configs.left_map1,
                               camera_configs.left_map2, cv2.INTER_LINEAR)

    img2_rectified = cv2.remap(imgR, camera_configs.right_map1,
                               camera_configs.right_map2, cv2.INTER_LINEAR)

    # trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos('num(x16)', 'depth') * 16
    if num < 16:
        num = 16
    blockSize = cv2.getTrackbarPos('blockSize', 'depth')
    if blockSize % 2 == 0:  # 防止blockSize取到偶数
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # 根据SGBM方法生成差异图
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num,
        blockSize=blockSize,
        uniquenessRatio=1,
        speckleRange=2,
        speckleWindowSize=50,
        disp12MaxDiff=200,
        P1=600,
        P2=2400,
        mode=cv2.STEREO_SGBM_MODE_HH)
    disparity = stereo.compute(img1_rectified, img2_rectified)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化函数算法
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity, camera_configs.Q,
                                    handleMissingValues=True) * 16
    cv2.setMouseCallback('depth', callback, threeD)

    cv2.imshow('left', img1_rectified)
    cv2.imshow('right', img2_rectified)
    cv2.imshow('depth', disp)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        if cv2.imwrite('shot/SGBM_rectifiedLeft.jpg', img1_rectified) \
         and cv2.imwrite('shot/SGBM_rectifiedRight.jpg', img2_rectified) \
         and cv2.imwrite('shot/SGBM_depth.jpg', disp) \
         and cv2.imwrite('shot/SGBM_frameL.jpg', left_frame) \
         and cv2.imwrite('shot/SGBM_frameR.jpg', right_frame):
            print('SnapShot saved.')

camera.release()
cv2.destroyAllWindows()

