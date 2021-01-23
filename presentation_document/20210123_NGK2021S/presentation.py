#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eel
import cv2 as cv
import csv
import base64
import copy

import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from model.deeplab_v3 import DeepLabV3
from model.pymediapipe import PyMediaPipe
from model.efficientnet_b0 import EfficientNetB0
from model.efficientdet_fingerframe import FingerFrameDetection


def demo01(deeplab_v3, image, prev_frame, color=[[0, 0, 0], [0, 0, 0]]):
    segmentation_map = deeplab_v3(image)
    demo_image1 = deeplab_v3.transparent_draw(image, segmentation_map,
                                              color[0], prev_frame)
    demo_image2 = deeplab_v3.transparent_draw(image, segmentation_map,
                                              color[1], prev_frame)
    return demo_image1, demo_image2


def demo02(efficientnet_b0, frame):
    classifications = efficientnet_b0(frame)
    demo_image = efficientnet_b0.draw1(frame, classifications)
    return demo_image


def demo03(efficientnet_b0, frame):
    classifications = efficientnet_b0(frame)
    demo_image = efficientnet_b0.draw2(frame, classifications)
    return demo_image


def demo04(pymediapipe, image):
    face_results, hand_results = pymediapipe(image)
    demo_image = pymediapipe.draw1(image, face_results, hand_results)
    return demo_image


def demo05(pymediapipe, image):
    face_results, hand_results = pymediapipe(image)
    demo_image = pymediapipe.draw2(image, face_results, hand_results)
    return demo_image


def demo06(deeplab_v3, image, prev_frame, color=[[0, 0, 0], [0, 0, 0]]):
    segmentation_map = deeplab_v3(image)
    demo_image1 = deeplab_v3.transparent_draw(image, segmentation_map,
                                              color[0], prev_frame)
    demo_image2 = deeplab_v3.transparent_draw(image, segmentation_map,
                                              color[1], prev_frame)
    return demo_image1, demo_image2


def demo07(pymediapipe, image):
    face_results, hand_results = pymediapipe(image)
    demo_image = pymediapipe.draw3(image, face_results, hand_results)
    return demo_image


# メイン処理 #############################################################
@eel.expose
def slide_change_event(val):
    global current_slide
    current_slide = val


# WebSlides側の頁数保持用変数 #############################################
global current_slide
current_slide = 1

# カメラ起動 #############################################################
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

ret, dummy_image = cap.read()
if not ret:
    exit()

# モデルロード ###########################################################
# DeepLabV3
deeplab_v3 = DeepLabV3()
_ = deeplab_v3(dummy_image)
# MediaPipe
pymediapipe = PyMediaPipe()
# EfficientNetB0
efficientnet_b0 = EfficientNetB0()
_ = efficientnet_b0(dummy_image)
# FingerFrameDetection
# fingerframe_detecter = FingerFrameDetection()
# _ = fingerframe_detecter(dummy_image)

# Eel起動 ###############################################################
eel.init('web')
eel.start('index.html',
          mode='chrome',
          cmdline_args=['--start-fullscreen'],
          block=False)

prev_frame = None
while True:
    eel.sleep(0.01)

    # カメラキャプチャ ###################################################
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv.flip(frame, 1)  # ミラー表示

    frame1, frame2 = frame, None
    if current_slide == 3:
        prev_frame = copy.deepcopy(frame)
    if current_slide == 4:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 5:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 6:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 7:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 8:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 9:
        color_list = [[255, 0, 255], [255, 0, 0]]
        frame1, frame2 = demo01(deeplab_v3,
                                frame,
                                prev_frame,
                                color=color_list)
    if current_slide == 18:
        frame1 = demo02(efficientnet_b0, frame)
    if current_slide == 20:
        frame1 = demo03(efficientnet_b0, frame)
    if current_slide == 22:
        frame1 = demo04(pymediapipe, frame)
    if current_slide == 24:
        frame1 = demo05(pymediapipe, frame)
    if current_slide == 26:
        color_list = [[0, 255, 0], [255, 0, 0]]
        frame1, _ = demo06(deeplab_v3, frame, prev_frame, color=color_list)
    if current_slide == 28:
        frame1 = demo07(pymediapipe, frame)

    # UI側へ転送
    _, imencode_image = cv.imencode('.jpg', frame1)
    base64_image = base64.b64encode(imencode_image)
    eel.set_base64image1("data:image/jpg;base64," +
                         base64_image.decode("ascii"))

    if frame2 is not None:
        _, imencode_image = cv.imencode('.jpg', frame2)
        base64_image = base64.b64encode(imencode_image)
        eel.set_base64image2("data:image/jpg;base64," +
                             base64_image.decode("ascii"))

    key = cv.waitKey(1)
    if key == 27:  # ESC
        break
