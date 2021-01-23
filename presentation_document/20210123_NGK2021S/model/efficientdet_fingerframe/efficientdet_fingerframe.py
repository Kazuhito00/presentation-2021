#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
from collections import deque

import cv2 as cv
import numpy as np
import tensorflow as tf2

from utils import CvOverlayImage


class FingerFrameDetection(object):
    def __init__(self):
        tf2.compat.v1.disable_eager_execution()

        # モデルロード ############################################################
        model_path = 'model/efficientdet_fingerframe/saved_model'

        DEFAULT_FUNCTION_KEY = 'serving_default'
        loaded_model = tf2.saved_model.load(model_path)
        self._inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

        buffer_len = 5
        self._deque_x1 = deque(maxlen=buffer_len)
        self._deque_y1 = deque(maxlen=buffer_len)
        self._deque_x2 = deque(maxlen=buffer_len)
        self._deque_y2 = deque(maxlen=buffer_len)

        self._video = cv.VideoCapture('model/efficientdet_fingerframe/map.mp4')
        self._score_th = 0.5

    def __call__(self, image):
        # image_width, image_height = image.shape[1], image.shape[0]

        # 検出実施 ############################################################
        frame = image[:, :, [2, 1, 0]]  # BGR2RGB
        image_np_expanded = np.expand_dims(frame, axis=0)

        output = self._run_inference_single_image(image_np_expanded)

        return output

    def _run_inference_single_image(self, image):
        tensor = tf2.convert_to_tensor(image)
        output = self._inference_func(tensor)
        print(output)

        output['num_detections'] = output['num_detections'][0]
        output['detection_classes'] = output['detection_classes'][0].numpy()
        output['detection_boxes'] = output['detection_boxes'][0].numpy()
        output['detection_scores'] = output['detection_scores'][0].numpy()
        return output

    def draw(self, image, result):
        image_width, image_height = image.shape[1], image.shape[0]

        num_detections = result['num_detections']
        for i in range(num_detections):
            score = result['detection_scores'][i]
            bbox = result['detection_boxes'][i]
            # class_id = output['detection_classes'][i].astype(np.int)

            if score < self._score_th:
                continue

            # 検出結果可視化 ###################################################
            x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
            x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)

            risize_ratio = 0.15
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            x1 = x1 + int(bbox_width * risize_ratio)
            y1 = y1 + int(bbox_height * risize_ratio)
            x2 = x2 - int(bbox_width * risize_ratio)
            y2 = y2 - int(bbox_height * risize_ratio)

            x1 = int((x1 - 5) / 10) * 10
            y1 = int((y1 - 5) / 10) * 10
            x2 = int((x2 + 5) / 10) * 10
            y2 = int((y2 + 5) / 10) * 10

            self._deque_x1.append(x1)
            self._deque_y1.append(y1)
            self._deque_x2.append(x2)
            self._deque_y2.append(y2)
            x1 = int(sum(self._deque_x1) / len(self._deque_x1))
            y1 = int(sum(self._deque_y1) / len(self._deque_y1))
            x2 = int(sum(self._deque_x2) / len(self._deque_x2))
            y2 = int(sum(self._deque_y2) / len(self._deque_y2))

            ret, video_frame = self._video.read()
            if ret is not False:
                self._video.grab()
                self._video.grab()

                debug_add_image = np.zeros((image_height, image_width, 3),
                                           np.uint8)
                map_resize_image = cv.resize(video_frame,
                                             ((x2 - x1), (y2 - y1)))
                debug_add_image = CvOverlayImage.overlay(
                    debug_add_image, map_resize_image, (x1, y1))
                debug_add_image = cv.cvtColor(debug_add_image,
                                              cv.COLOR_BGRA2BGR)
                # cv.imshow('1', debug_add_image)
                debug_image = cv.addWeighted(image, 1.0, debug_add_image, 2.0,
                                             0)
            else:
                self._video = cv.VideoCapture('map.mp4')

        return debug_image
