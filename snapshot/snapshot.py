# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年05月17日
"""
from snapshot_depth import camera_configs
import cv2

cv2.namedWindow('left')
cv2.namedWindow('right')
cv2.moveWindow('left', 0, 0)
cv2.moveWindow('right', 800, 0)
camera = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

# 设置分辨率，左右的摄像机同一频率，同一设备ID
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        if cv2.imwrite('shot/rectifiedLeft.jpg', img1_rectified) \
         and cv2.imwrite('shot/rectifiedRight.jpg', img2_rectified) \
         and cv2.imwrite('shot/frameL.jpg', left_frame) \
         and cv2.imwrite('shot/frameR.jpg', right_frame):
            print('SnapShot saved.')

camera.release()
cv2.destroyAllWindows()
