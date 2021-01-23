#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import cv2 as cv
import numpy as np
import tensorflow as tf2

import utils.cvui as cvui
from utils import CvDrawText


class EfficientNetB0(object):
    def __init__(self):
        # ImageNet 日本語ラベル ###################################################
        jsonfile = open('utils/imagenet_class_index.json',
                        'r',
                        encoding="utf-8_sig")
        imagenet_ja_labels = json.load(jsonfile)
        self._imagenet_ja_label = {}
        for label_temp in imagenet_ja_labels:
            self._imagenet_ja_label[label_temp['num']] = label_temp['ja']

        # モデルロード ############################################################
        self._model = tf2.keras.applications.EfficientNetB0(
            include_top=True,
            weights='imagenet',
            input_shape=(224, 224, 3),
        )

        self._detection_count = 0

    def __call__(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # 検出実施 ############################################################
        x_offset = -160
        y_offset = 10
        self._trim_x1 = int(image_width / 4) + 80 + x_offset
        self._trim_x2 = int(image_width / 4 * 3) - 80 + x_offset
        self._trim_y1 = int(image_height / 5) + y_offset
        self._trim_y2 = int(image_height / 5 * 4) + y_offset
        trimming_image = copy.deepcopy(image[self._trim_y1:self._trim_y2,
                                             self._trim_x1:self._trim_x2])

        classifications = self._run_classify(trimming_image)

        return classifications

    def draw1(self, image, classifications):
        # image_width, image_height = image.shape[1], image.shape[0]

        # 表示名作成
        classification_string = ""
        for classification in classifications:
            if float(classification[2]) > 0.5:
                self._detection_count += 1

                classification_string = self._imagenet_ja_label[
                    classification[0]] + ":" + str('{:.1f}'.format(
                        float(classification[2]) * 100)) + "%"
            else:
                self._detection_count = 0
            break  # 1件のみ

        # 描画
        debug_image = self._draw_demo_image(
            image,
            self._detection_count,
            classification_string,
            (self._trim_x1, self._trim_y1, self._trim_x2, self._trim_y2),
        )

        return debug_image

    def draw2(self, image, classifications):
        image_width, image_height = image.shape[1], image.shape[0]

        cvuiframe = np.zeros((image_height + 6, image_width + 6 + 200, 3),
                             np.uint8)
        cvuiframe[:] = (49, 52, 49)

        # 画像：撮影映像
        display_frame = copy.deepcopy(image[:, 0:image_width - 130])
        cvui.image(cvuiframe, 3, 3, display_frame)

        # 文字列、バー：クラス分類結果
        if classifications is not None:
            for i, classification in enumerate(classifications):
                cvui.printf(cvuiframe, image_width - 100,
                            int(image_height / 3) + (i * 40), 0.4, 0xFFFFFF,
                            classification[1])
                cvui.rect(cvuiframe, image_width - 100,
                          int(image_height / 3) + 15 + (i * 40),
                          int(181 * float(classification[2])), 12, 0xFFFFFF,
                          0xFFFFFF)

        return cvuiframe

    def _run_classify(self, image, top_num=5):
        """
        [summary]
            画像クラス分類
        Parameters
        ----------
        model : model
            クラス分類用モデル
        image : image
            推論対象の画像
        None
        """
        inp = cv.resize(image, (224, 224))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        inp = np.expand_dims(inp, axis=0)
        tensor = tf2.convert_to_tensor(inp)
        tensor = tf2.keras.applications.efficientnet.preprocess_input(tensor)

        classifications = self._model.predict(tensor)

        classifications = tf2.keras.applications.efficientnet.decode_predictions(
            classifications,
            top=top_num,
        )
        classifications = np.squeeze(classifications)
        return classifications

    def _draw_demo_image(
        self,
        image,
        detection_count,
        classification_string,
        trim_point,
    ):
        image_width = int((trim_point[0] + trim_point[2]))
        image_height = int((trim_point[1] + trim_point[3]))

        # フォント
        font_path = './utils/font/KosugiMaru-Regular.ttf'

        # 四隅枠表示
        if detection_count < 4:
            gap_length = 300  # int((trim_point[2] - trim_point[0]) / 20) * 19
            cv.line(image, (trim_point[0], trim_point[1]),
                    (trim_point[2] - gap_length, trim_point[1]),
                    (255, 255, 255), 3)
            cv.line(image, (trim_point[0] + gap_length, trim_point[1]),
                    (trim_point[2], trim_point[1]), (255, 255, 255), 3)
            cv.line(image, (trim_point[2], trim_point[1]),
                    (trim_point[2], trim_point[3] - gap_length),
                    (255, 255, 255), 3)
            cv.line(image, (trim_point[2], trim_point[1] + gap_length),
                    (trim_point[2], trim_point[3]), (255, 255, 255), 3)
            cv.line(image, (trim_point[0], trim_point[3]),
                    (trim_point[2] - gap_length, trim_point[3]),
                    (255, 255, 255), 3)
            cv.line(image, (trim_point[0] + gap_length, trim_point[3]),
                    (trim_point[2], trim_point[3]), (255, 255, 255), 3)
            cv.line(image, (trim_point[0], trim_point[1]),
                    (trim_point[0], trim_point[3] - gap_length),
                    (255, 255, 255), 3)
            cv.line(image, (trim_point[0], trim_point[1] + gap_length),
                    (trim_point[0], trim_point[3]), (255, 255, 255), 3)

        line_x1 = int(image_width / 1.55)
        line_x2 = int(image_width / 1.1)
        line_y = int(image_height / 4)

        # 回転丸表示
        if detection_count > 0:
            draw_angle = int(detection_count * 45)
            cv.ellipse(image, (int(image_width / 2), int(image_height / 2)),
                       (10, 10), -45, 0, draw_angle, (255, 255, 255), -1)
        # 斜線表示
        if detection_count > 10:
            cv.line(image, (int(image_width / 2), int(image_height / 2)),
                    (line_x1, line_y), (255, 255, 255), 2)

        # 横線・分類名・スコア表示
        if detection_count > 10:
            font_size = 32
            cv.line(image, (line_x1, line_y), (line_x2, line_y),
                    (255, 255, 255), 2)
            image = CvDrawText.puttext(
                image, classification_string,
                (line_x1 + 10, line_y - int(font_size * 1.25)), font_path,
                font_size, (255, 255, 255))

        return image
