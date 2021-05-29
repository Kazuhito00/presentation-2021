#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eel
import cv2 as cv
import csv
import base64
import copy

import numpy as np

from model.pymediapipe import PyMediaPipe


def demo01(pymediapipe, image):
    face_detection_results = pymediapipe.process_face_detection(image)
    demo_image = pymediapipe.draw01(image, face_detection_results)
    return demo_image


def demo02(pymediapipe, image):
    face_mesh_results = pymediapipe.process_face_mesh(image)
    demo_image = pymediapipe.draw02(image, face_mesh_results)
    return demo_image


def demo03(pymediapipe, image):
    pose_results = pymediapipe.process_pose(image)
    demo_image = pymediapipe.draw03(image, pose_results)
    return demo_image


def demo04(pymediapipe, image):
    pose_results = pymediapipe.process_hands(image)
    demo_image = pymediapipe.draw04(image, pose_results)
    return demo_image


def demo05(pymediapipe, image):
    holistic_results = pymediapipe.process_holistic(image)
    demo_image = pymediapipe.draw05(image, holistic_results)
    return demo_image


def demo06(pymediapipe, image):
    objectron_results = pymediapipe.process_objectron(image)
    demo_image = pymediapipe.draw06(image, objectron_results)
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
# MediaPipe
pymediapipe = PyMediaPipe()

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
    if current_slide == 18:
        frame1 = demo01(pymediapipe, frame)
    if current_slide == 20:
        frame1 = demo02(pymediapipe, frame)
    if current_slide == 22:
        frame1 = demo03(pymediapipe, frame)
    if current_slide == 24:
        frame1 = demo04(pymediapipe, frame)
    if current_slide == 26:
        frame1 = demo05(pymediapipe, frame)
    if current_slide == 28:
        frame1 = demo06(pymediapipe, frame)

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
