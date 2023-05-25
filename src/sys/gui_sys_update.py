import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget, QApplication

import json
import time
import re

# 导入python-opencv和numpy模块
import cv2
import numpy as np

import matrices

# 相机畸变参数，经畸变矫正得来
MTX_CAMERA1 = np.float32([[227.21233652, 0, 281.86422286], [0, 224.58459497, 210.10036974], [0, 0, 1]])  # 自己的 7
DIST_CAMERA1 = np.float32([-0.30034003, 0.07065534, -0.00831145, 0.01151889, -0.00699896])

MTX_CAMERA2 = np.float32([[197.8478, -1.7711, 302.2921], [0, 195.1688, 187.6430], [0, 0, 1]])
DIST_CAMERA2 = np.float32([-0.2805, 0.0646, -0.0059, 0.0024, -0.0025])

# 定义棋盘格转换后在图片中的边长，单位为像素
BLOCK_SIZE = 25  # mby# 重要比例参数，用于调整最终成像分辨率，会影响图中区域的比例

VIDEO_WIDTH = 500
VIDEO_HEIGHT = 700
VIDEOSRC = 0
TOTAL_SIZE_BLOCKS = [20, 28]
FULL_RESOLUTION = (TOTAL_SIZE_BLOCKS[0] * BLOCK_SIZE, TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE)
FRAME = np.zeros([FULL_RESOLUTION[1], FULL_RESOLUTION[0], 3], np.uint8)

#  定义分块大小
TOP_BLOCKS = [20, 6]
LEFT_BLOCKS = [5, 18]
RIGHT_BLOCKS = [5, 18]
BOTTOM_LEFT_BLOCKS = [10, 5]
BOTTOM_RIGHT_BLOCKS = [10, 5]
TOTAL_SIZE_BLOCKS = [20, 28]

FULL_RESOLUTION = (TOTAL_SIZE_BLOCKS[0] * BLOCK_SIZE, TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE)

top_edge = [100, 1, 200, 2]
left_edge = [100, 1, 200, 2]
bottom_edge = [100, 1, 200, 2]
right_edge = [100, 1, 200, 2]

cap_device0 = cv2.VideoCapture('device0.avi')
cap_device5 = cv2.VideoCapture('device5.avi')

CAMERA_STATUS = [True, True, True, True, True]


def stitch_process():
    now = time.time()
    # 全景拼接图片帧变量
    global FRAME

    # 图像读取与初步处理部分
    # img_device0 = cv2.imread("./chessboard_images/img_device0_around.jpg")
    # img_device5 = cv2.imread("./chessboard_images/img_device5_head.jpg")
    ret1, img_device0 = cap_device0.read()
    if not ret1:
        # 如果读取到了最后一帧，则重置文件指针到开头再次循环播放
        cap_device0.set(cv2.CAP_PROP_POS_FRAMES, 0)
        img_device0 = cap_device0.read()
    # cv2.imshow("?",img_device0)
    ret2, img_device5 = cap_device5.read()
    if not ret2:
        # 如果读取到了最后一帧，则重置文件指针到开头再次循环播放
        cap_device5.set(cv2.CAP_PROP_POS_FRAMES, 0)
        img_device5 = cap_device5.read()
    height, width, _ = img_device0.shape
    h, w = height // 2, width // 2
    # 原生全景图
    top_raw = img_device5  # mby# cv2.imread()用于读取图片文件 默认加载RGB 可仅加载灰度图片
    bottom_left_raw = img_device0[0:h, w:2 * w]
    bottom_right_raw = img_device0[h:2 * h, w:2 * w]
    left_raw = img_device0[0:h, 0:w]
    right_raw = img_device0[h:2 * h, 0:w]
    # RIGHT_TEST = cv2.imread("./chessboard_images/test-test.jpg")
    # print(left_raw.shape)
    # cv2.imshow("weqwe", bottom_left_raw)

    top_resized = cv2.resize(top_raw, (640, 360), interpolation=cv2.INTER_AREA)
    left_resized = cv2.resize(left_raw, (640, 360), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_raw, (640, 360), interpolation=cv2.INTER_AREA)
    bottom_left_resized = cv2.resize(bottom_left_raw, (640, 360), interpolation=cv2.INTER_AREA)
    bottom_right_resized = cv2.resize(bottom_right_raw, (640, 360), interpolation=cv2.INTER_AREA)

    # cv2.waitKey(0)

    # 畸变矫正
    top_undistorted = cv2.undistort(top_resized, MTX_CAMERA2, DIST_CAMERA2, None, None)
    left_undistorted = cv2.undistort(left_resized, MTX_CAMERA2, DIST_CAMERA2, None, None)
    right_undistorted = cv2.undistort(right_resized, MTX_CAMERA2, DIST_CAMERA2, None, None)
    bottom_left_undistorted = cv2.undistort(bottom_left_resized, MTX_CAMERA2, DIST_CAMERA2, None, None)
    bottom_right_undistorted = cv2.undistort(bottom_right_resized, MTX_CAMERA1, DIST_CAMERA1, None, None)
    print("预处理所花费的时间" + str(time.time() - now))

    # cv2.imshow("mine", bottom_right_undistorted)
    # cv2.imshow("weqw1", bottom_left_undistorted)

    # cv2.waitKey(0)

    top_resized1 = cv2.resize(top_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
    for j in range(0, TOP_BLOCKS[0], 1):
        result = cv2.warpPerspective(top_resized1, matrices.top_homology[j], FULL_RESOLUTION)
        # print(matrices.test_h[j])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        FRAME[0:TOP_BLOCKS[1] * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = result[0:TOP_BLOCKS[1] * BLOCK_SIZE,
                                                                                   j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
    # test_result = FRAME[0:TOP_BLOCKS[1] * BLOCK_SIZE, 0:TOP_BLOCKS[0] * BLOCK_SIZE]
    # cv2.imshow("TOP_result", test_result)
    # print("TOP一帧所花费的时间" + str(time.time() - now))
    # cv2.waitKey(0)

    left_resized1 = cv2.resize(left_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
    for j in range(0, LEFT_BLOCKS[1], 1):
        result = cv2.warpPerspective(left_resized1, matrices.left_homology[j], FULL_RESOLUTION)
        # print(matrices.test_h[j])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        FRAME[(j + TOP_BLOCKS[1]) * BLOCK_SIZE:(j + TOP_BLOCKS[1] + 1) * BLOCK_SIZE,
        0:LEFT_BLOCKS[0] * BLOCK_SIZE] = result[(j + TOP_BLOCKS[1]) * BLOCK_SIZE:(j + TOP_BLOCKS[1] + 1) * BLOCK_SIZE,
                                         0:LEFT_BLOCKS[0] * BLOCK_SIZE]
    # test_result = FRAME[TOP_BLOCKS[1] * BLOCK_SIZE:(TOP_BLOCKS[1] + LEFT_BLOCKS[1]) * BLOCK_SIZE,
    #               0:LEFT_BLOCKS[0] * BLOCK_SIZE]
    # cv2.imshow("TOP_result", test_result)
    # print("LEFT一帧所花费的时间" + str(time.time() - now))
    # cv2.waitKey(0)

    right_resized1 = cv2.resize(right_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
    for j in range(0, RIGHT_BLOCKS[1], 1):
        result = cv2.warpPerspective(right_resized1, matrices.right_homology[j], FULL_RESOLUTION)
        # print(matrices.test_h[j])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        FRAME[(j + TOP_BLOCKS[1]) * BLOCK_SIZE:(j + TOP_BLOCKS[1] + 1) * BLOCK_SIZE,
        (TOTAL_SIZE_BLOCKS[0] - RIGHT_BLOCKS[0]) * BLOCK_SIZE: TOTAL_SIZE_BLOCKS[0] * BLOCK_SIZE] = result[(j +
                                                                                                            TOP_BLOCKS[
                                                                                                                1]) * BLOCK_SIZE:(
                                                                                                                                         j +
                                                                                                                                         TOP_BLOCKS[
                                                                                                                                             1] + 1) * BLOCK_SIZE,
                                                                                                    (TOTAL_SIZE_BLOCKS[
                                                                                                         0] -
                                                                                                     RIGHT_BLOCKS[
                                                                                                         0]) * BLOCK_SIZE:
                                                                                                    TOTAL_SIZE_BLOCKS[
                                                                                                        0] * BLOCK_SIZE]
    test_result = FRAME[TOP_BLOCKS[1] * BLOCK_SIZE:(RIGHT_BLOCKS[1] + TOP_BLOCKS[1]) * BLOCK_SIZE,
                  (TOTAL_SIZE_BLOCKS[0] - RIGHT_BLOCKS[0]) * BLOCK_SIZE: TOTAL_SIZE_BLOCKS[0] * BLOCK_SIZE]
    # cv2.imshow("TOP_result", test_result)
    # print("TOP一帧所花费的时间" + str(time.time() - now))
    # cv2.waitKey(0)

    bottom_left_resized1 = cv2.resize(bottom_left_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
    for j in range(0, BOTTOM_LEFT_BLOCKS[0], 1):
        result = cv2.warpPerspective(bottom_left_resized1, matrices.bottom_left_homology[j], FULL_RESOLUTION)
        # print(matrices.test_h[j])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        FRAME[(TOTAL_SIZE_BLOCKS[1] - BOTTOM_LEFT_BLOCKS[1]) * BLOCK_SIZE:TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE,
        j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = result[(TOTAL_SIZE_BLOCKS[1] - BOTTOM_LEFT_BLOCKS[1]) * BLOCK_SIZE:
                                                      TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE,
                                               j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
    # test_result = FRAME[(TOTAL_SIZE_BLOCKS[1] - BOTTOM_LEFT_BLOCKS[1]) * BLOCK_SIZE:TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE,
    #               0:BOTTOM_LEFT_BLOCKS[0] * BLOCK_SIZE]
    # cv2.imshow("BOTTOM_LEFT_result", test_result)
    # print("TOP一帧所花费的时间" + str(time.time() - now))
    # cv2.waitKey(0)

    bottom_right_resized1 = cv2.resize(bottom_right_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
    for j in range(0, BOTTOM_RIGHT_BLOCKS[0], 1):
        result = cv2.warpPerspective(bottom_right_resized1, matrices.bottom_right_homology[j], FULL_RESOLUTION)
        # print(matrices.test_h[j])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        FRAME[(TOTAL_SIZE_BLOCKS[1] - BOTTOM_RIGHT_BLOCKS[1]) * BLOCK_SIZE:TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE,
        (BOTTOM_LEFT_BLOCKS[0] + j) * BLOCK_SIZE:(BOTTOM_LEFT_BLOCKS[
                                                      0] + j + 1) * BLOCK_SIZE] = result[(TOTAL_SIZE_BLOCKS[1] -
                                                                                          BOTTOM_RIGHT_BLOCKS[
                                                                                              1]) * BLOCK_SIZE:
                                                                                         TOTAL_SIZE_BLOCKS[
                                                                                             1] * BLOCK_SIZE,
                                                                                  (BOTTOM_LEFT_BLOCKS[
                                                                                       0] + j) * BLOCK_SIZE:(
                                                                                                                    BOTTOM_LEFT_BLOCKS[
                                                                                                                        0] + j + 1) * BLOCK_SIZE]
    # test_result = FRAME[(TOTAL_SIZE_BLOCKS[1] - BOTTOM_RIGHT_BLOCKS[1]) * BLOCK_SIZE:TOTAL_SIZE_BLOCKS[1] * BLOCK_SIZE,
    #               BOTTOM_LEFT_BLOCKS[0] * BLOCK_SIZE:TOTAL_SIZE_BLOCKS[0] * BLOCK_SIZE]
    # cv2.imshow("TOP_result", test_result)
    # print("TOP一帧所花费的时间" + str(time.time() - now))
    # cv2.waitKey(0)

    # 图像拼接
    # FRAME[0:40 * BLOCK_SIZE, 0:LEFT_WIDTH] = LEFT_IMG[0:40 * BLOCK_SIZE, 0:LEFT_WIDTH]
    # FRAME[0:TOP_HEIGHT, 0:30 * BLOCK_SIZE] = TOP_IMG[0:TOP_HEIGHT, 0:40 * BLOCK_SIZE]
    # FRAME[40 * BLOCK_SIZE - BOTTOM_HEIGHT:40 * BLOCK_SIZE, 0:30 * BLOCK_SIZE] = BOTTOM_IMG[
    #                                                                             40 * BLOCK_SIZE - BOTTOM_HEIGHT:40 * BLOCK_SIZE,
    #                                                                             0:30 * BLOCK_SIZE]
    # FRAME[TOP_HEIGHT:40 * BLOCK_SIZE - BOTTOM_HEIGHT, 30 * BLOCK_SIZE - RIGHT_WIDTH:30 * BLOCK_SIZE] = RIGHT_IMAGE[
    #                                                                                                    TOP_HEIGHT:40 * BLOCK_SIZE - BOTTOM_HEIGHT,
    #                                                                                                    30 * BLOCK_SIZE - RIGHT_WIDTH:30 * BLOCK_SIZE]
    # cv2.imshow('result', FRAME)  # 按照像素点比例拼接图片#

    # cv2.imwrite("pinjie.jpg", expand)
    noww = str(time.time() - now)
    print("处理一帧所花费的时间" + noww)
    # cv2.waitKey(10)  # mby# waitKey()–是在一个给定的时间内(单位ms)等待用户按键触发; 如果用户没有按下键,则继续等待 (循环)


# if __name__ == '__main__':
#     while True:
#         stitch_process()

# 有用的测试testdemo
# right_resized = cv2.resize(right_undistorted, (1280, 720), interpolation=cv2.INTER_AREA)
#
#     for j in range(0, 19, 1):
#         result = cv2.warpPerspective(right_resized, matrices.test_h[j], FULL_RESOLUTION)
#         # print(matrices.test_h[j])
#         # cv2.imshow("result", result)
#         # cv2.waitKey(0)
#         FRAME[0:8 * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = result[0:8 * BLOCK_SIZE,
#                                                                        j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
#     test_result = FRAME[0:8 * BLOCK_SIZE, 0:19 * BLOCK_SIZE]
#     cv2.imshow("result", test_result)
#     print("一帧所花费的时间" + str(time.time() - now))
#     cv2.waitKey(0)
#     exit()


from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QLabel


class VideoDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # 创建 QLabel 并初始化宽度和高度
        self.video_label = QLabel(self)
        self.video_label.resize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setVisible(False)
        # 创建 QVBoxLayout 布局组件并将其添加到 QWidget 上
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)

        # 创建一个 QPushButton 对象，并设置它的文本为 “显示图像”
        self.button = QPushButton('显示图像', self)
        self.button.clicked.connect(self.show_image)  # 连接按钮的点击事件到 show_image 槽函数上
        self.layout.addWidget(self.button)

        # 创建 QLabel 组件来显示五个布尔变量的状态
        self.status_labels = [QLabel(str(i), self) for i in range(5)]
        for status_label in self.status_labels:
            self.layout.addWidget(status_label)

        # 设置 QFont 来使得字体更加清晰易读
        font = QFont()
        font.setPointSize(12)
        for status_label in self.status_labels:
            status_label.setFont(font)

        # 将布局组件设置到主窗口中
        self.setLayout(self.layout)

        # 初始化 OpenCV VideoCapture 对象
        # self.cap = cv2.VideoCapture(VIDEOSRC)

        # 创建 QTimer 并连接到更新函数
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)  # 30 FPS

        # 布尔变量
        self.status = [False] * 5

    def update_image(self):
        # 从 VideoCapture 中读取帧

        # 将 BGR 帧转换为 RGB 和 QImage 格式

        self.set_status()  # 更新布尔变量状态
        stitch_process()
        h, w, ch = FRAME.shape
        qimg = QImage(FRAME, w, h, ch * w, QImage.Format_RGB888)

        # 调整 QLabel 和 QPixmap 大小以匹配视频流尺寸
        self.video_label.setPixmap(QPixmap(qimg).scaled(VIDEO_WIDTH, VIDEO_HEIGHT))

    def show_image(self):
        self.video_label.setVisible(True)  # 显示视频图像
        self.button.setVisible(False)  # 隐藏按钮

    def set_status(self):
        # 设置布尔变量的状态
        for i in range(5):
            self.status[i] = CAMERA_STATUS[i]  # 这里只是一个示例，你需要根据具体需求修改实现方式

        # 更新布尔变量的值到 QLabel 组件上
        for i, status_label in enumerate(self.status_labels):
            state_str = '正常' if self.status[i] else '失去信号'  # 利用三目运算符将 True/False 转化为 中文‘真’/‘假’
            status_str = f"相机 {i + 1} 状态: {state_str}"
            status_label.setText(status_str)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    display = VideoDisplay()
    display.show()
    sys.exit(app.exec_())
