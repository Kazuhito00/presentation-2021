#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import copy
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvOverlayImage


class PyMediaPipe(object):
    def __init__(self, config=None):
        if config is None:
            self._config = {
                'face_mesh_static_image_mode': False,
                'face_mesh_max_num_faces': 1,
                'face_mesh_min_detection_confidence': 0.5,
                'face_mesh_min_tracking_confidence': 0.5,
                'hands_static_image_mode': False,
                'hands_max_num_hands': 1,
                'hands_min_detection_confidence': 0.7,
                'hands_min_tracking_confidence': 0.5,
                'pose_static_image_mode': False,
                'pose_upper_body_only': False,
                'pose_smooth_landmarks': True,
                'pose_min_detection_confidence': 0.5,
                'pose_min_tracking_confidence': 0.5,
                'holistic_static_image_mode': False,
                'holistic_upper_body_only': False,
                'holistic_smooth_landmarks': True,
                'holistic_min_detection_confidence': 0.5,
                'holistic_min_tracking_confidence': 0.5,
            }
        else:
            self._config = config

        self.hand = self._create_hands_solution()
        self.face = self._create_face_mesh_solution()

        self._image_pathlist = sorted(
            glob.glob('model/pymediapipe/image/*.png'))
        self._images = []
        for image_path in self._image_pathlist:
            self._images.append(cv.imread(image_path, cv.IMREAD_UNCHANGED))

        self._animation_counter = 0

        max_history = 3
        self._face_x1 = deque(maxlen=max_history)
        self._face_y1 = deque(maxlen=max_history)
        self._face_x2 = deque(maxlen=max_history)
        self._face_y2 = deque(maxlen=max_history)

        self._ceil_num = 50
        self._image_ratio = 1.2

        self._pointer_x = 0

    def __call__(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        hand_results = self.hand.process(rgb_image)
        face_results = self.face.process(rgb_image)
        rgb_image.flags.writeable = True

        return face_results, hand_results

    def draw1(self, image, face_results, hand_results):
        # 顔
        if face_results.multi_face_landmarks is not None:
            for face_landmarks in face_results.multi_face_landmarks:
                # 外接矩形の計算
                brect = self._calc_bounding_rect(image, face_landmarks)
                self._face_x1.append(brect[0])
                self._face_y1.append(brect[1])
                self._face_x2.append(brect[2])
                self._face_y2.append(brect[3])
        # 描画
        image = self._draw_face_image(image)
        return image

    def draw2(self, image, face_results, hand_results):
        # 手
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness):
                # 外接矩形の計算
                brect = self._calc_bounding_rect(image, hand_landmarks)
                # 描画
                # image = self._draw_landmarks(image, hand_landmarks, handedness)
                # image = self._draw_bounding_rect(image, brect)
                image = self._bba_rotate_dotted_ring3(
                    image,
                    (brect[0], brect[1]),
                    (brect[2], brect[3]),
                    animation_count=self._animation_counter,
                )
                self._animation_counter += 0.5
        return image

    def draw3(self, image, face_results, hand_results):
        image1 = copy.deepcopy(image)
        # 手
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness):
                # 描画
                _ = self._draw_landmarks(image, hand_landmarks, handedness)

        image2 = self._polygon_filter(image,
                                      akaze_threshold=0.0002,
                                      additional_points=[[100, 0], [200, 0]],
                                      draw_line=True)

        image1_width, image1_height = image1.shape[1], image1.shape[0]
        image2_width, image2_height = image2.shape[1], image2.shape[0]
        if ((image1_width != image2_width)
                or (image1_height != image2_height)):
            image2 = cv.resize(image2, (image1_width, image1_height))

        image_height = image1.shape[0]

        crop_image1 = image1[:, 0:self._pointer_x]
        crop_image2 = image2[:, self._pointer_x + 1:]
        concat_image = np.concatenate([crop_image1, crop_image2], axis=1)

        cv.line(concat_image, (self._pointer_x, 1),
                (self._pointer_x, image_height), (255, 255, 255),
                thickness=2)
        return concat_image

    def _create_holistic_solution(self):
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=self._config['holistic_static_image_mode'],
            upper_body_only=self._config['holistic_upper_body_only'],
            smooth_landmarks=self._config['holistic_smooth_landmarks'],
            min_detection_confidence=self.
            _config['holistic_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['holistic_min_tracking_confidence'],
        )
        return holistic

    def _create_face_mesh_solution(self):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self._config['face_mesh_static_image_mode'],
            max_num_faces=self._config['face_mesh_max_num_faces'],
            min_detection_confidence=self.
            _config['face_mesh_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['face_mesh_min_tracking_confidence'],
        )
        return face_mesh

    def _create_hands_solution(self):
        hands = mp.solutions.hands.Hands(
            static_image_mode=self._config['hands_static_image_mode'],
            max_num_hands=self._config['hands_max_num_hands'],
            min_detection_confidence=self.
            _config['hands_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['hands_min_tracking_confidence'],
        )
        return hands

    def _create_pose_solution(self):
        pose = mp.solutions.pose.Pose(
            static_image_mode=self._config['pose_static_image_mode'],
            upper_body_only=self._config['pose_upper_body_only'],
            smooth_landmarks=self._config['pose_smooth_landmarks'],
            min_detection_confidence=self.
            _config['pose_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['pose_min_tracking_confidence'],
        )
        return pose

    def _calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def _draw_face_image(self, image):
        face_x1, face_y1, face_x2, face_y2 = None, None, None, None
        if len(self._face_x1) > 0:
            face_x1 = int(sum(self._face_x1) / len(self._face_x1))
        if len(self._face_y1) > 0:
            face_y1 = int(sum(self._face_y1) / len(self._face_y1))
        if len(self._face_x2) > 0:
            face_x2 = int(sum(self._face_x2) / len(self._face_x2))
        if len(self._face_y2) > 0:
            face_y2 = int(sum(self._face_y2) / len(self._face_y2))
        if not (None in [face_x1, face_y1, face_x2, face_y2]):
            # 顔の立幅に合わせて重畳画像をリサイズ
            face_height, face_width = self._images[0].shape[:2]
            resize_ratio = (face_y2 - face_y1) / face_height
            resize_face_height = int(face_height * resize_ratio)
            resize_face_width = int(face_width * resize_ratio)

            resize_face_height = int(
                (resize_face_height +
                 (self._ceil_num - 1)) / self._ceil_num * self._ceil_num)
            resize_face_width = int(
                (resize_face_width +
                 (self._ceil_num - 1)) / self._ceil_num * self._ceil_num)
            resize_face_height = int(resize_face_height * self._image_ratio)
            resize_face_width = int(resize_face_width * self._image_ratio)

            resize_face = cv.resize(self._images[int(self._animation_counter)],
                                    (resize_face_width, resize_face_height))

            y_offset = int(-1 * (resize_face_height / 10))
            # 画像描画
            overlay_x = int(
                (face_x2 + face_x1) / 2) - int(resize_face_width / 2)
            overlay_y = int(
                (face_y2 + face_y1) / 2) - int(resize_face_height / 2)
            image = CvOverlayImage.overlay(image, resize_face,
                                           (overlay_x, overlay_y + y_offset))

            self._animation_counter += 0.5
            if self._animation_counter >= len(self._images):
                self._animation_counter = 0

        return image

    def _draw_landmarks(self, image, landmarks, handedness):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # キーポイント
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))

            if index == 0:  # 手首1
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 1:  # 手首2
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 125, 0), 2)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 8:  # 人差指：指先
                self._pointer_x = landmark_x
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 125, 0), 2)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 125, 0), 2)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 125, 0), 2)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 125, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 125, 0), 2)

        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv.line(image, landmark_point[2], landmark_point[3], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[3], landmark_point[4], (0, 125, 0),
                    2)

            # 人差指
            cv.line(image, landmark_point[5], landmark_point[6], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[6], landmark_point[7], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[7], landmark_point[8], (0, 125, 0),
                    2)

            # 中指
            cv.line(image, landmark_point[9], landmark_point[10], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[10], landmark_point[11], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[11], landmark_point[12], (0, 125, 0),
                    2)

            # 薬指
            cv.line(image, landmark_point[13], landmark_point[14], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[14], landmark_point[15], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[15], landmark_point[16], (0, 125, 0),
                    2)

            # 小指
            cv.line(image, landmark_point[17], landmark_point[18], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[18], landmark_point[19], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[19], landmark_point[20], (0, 125, 0),
                    2)

            # 手の平
            cv.line(image, landmark_point[0], landmark_point[1], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[1], landmark_point[2], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[2], landmark_point[5], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[5], landmark_point[9], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[9], landmark_point[13], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[13], landmark_point[17], (0, 125, 0),
                    2)
            cv.line(image, landmark_point[17], landmark_point[0], (0, 125, 0),
                    2)

        return image

    def _draw_bounding_rect(self, image, brect):
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 125, 0), 2)
        return image

    def _bba_rotate_dotted_ring3(
            self,
            image,
            p1,
            p2,
            color=(255, 255, 205),
            fps=10,
            animation_count=0,
    ):

        draw_image = copy.deepcopy(image)

        animation_count = int(135 / fps) * animation_count

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        radius = int((y2 - y1) * (5 / 10))
        ring_thickness = int(radius / 20)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 0 + animation_count, 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 80 + animation_count, 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 150 + animation_count, 0, 30,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 200 + animation_count, 0, 10,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 230 + animation_count, 0, 10,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 260 + animation_count, 0, 60,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 337 + animation_count, 0, 5,
                   color, ring_thickness)

        radius = int((y2 - y1) * (4.5 / 10))
        ring_thickness = int(radius / 10)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 0 - animation_count, 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 80 - animation_count, 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 150 - animation_count, 0, 30,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 200 - animation_count, 0, 30,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 260 - animation_count, 0, 60,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
            (y1 + y2) / 2)), (radius, radius), 337 - animation_count, 0, 5,
                   color, ring_thickness)

        radius = int((y2 - y1) * (4 / 10))
        ring_thickness = int(radius / 15)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 30 + int(animation_count / 3 * 2), 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 110 + int(animation_count / 3 * 2), 0, 50,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 180 + int(animation_count / 3 * 2), 0, 30,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 230 + int(animation_count / 3 * 2), 0, 10,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 260 + int(animation_count / 3 * 2), 0, 10,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 290 + int(animation_count / 3 * 2), 0, 60,
                   color, ring_thickness)
        cv.ellipse(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                   (radius, radius), 367 + int(animation_count / 3 * 2), 0, 5,
                   color, ring_thickness)

        return draw_image

    def _polygon_filter(self,
                        image,
                        akaze_threshold=0.00001,
                        additional_points=[],
                        draw_line=False):
        """ポリゴンフィルターを適用した画像を返す

        Args:
            image: OpenCV Image
            akaze_threshold: AKAZE Threshold
            additional_point: Subdiv2D Points for additional Insert
            draw_line: Whether to draw the sides of the triangle

        Returns:
            Image after applying the filter.
        """
        height, width, _ = image.shape[0], image.shape[1], image.shape[2]

        # 特徴点抽出
        akaze = cv.AKAZE_create(threshold=akaze_threshold)
        key_points, _ = akaze.detectAndCompute(image, None)
        key_points = cv.KeyPoint_convert(key_points)

        # ドロネー図作成
        subdiv = cv.Subdiv2D((0, 0, width, height))

        subdiv.insert((0, 0))
        subdiv.insert((width - 1, 0))
        subdiv.insert((0, height - 1))
        subdiv.insert((width - 1, height - 1))
        subdiv.insert((int(width / 2), 0))
        subdiv.insert((0, int(height / 2)))
        subdiv.insert((width - 1, int(height / 2)))
        subdiv.insert((int(width / 2), height - 1))
        subdiv.insert((int(width / 2), int(height / 2)))
        for key_point in key_points:
            subdiv.insert((int(key_point[0]), int(key_point[1])))
        for additional_point in additional_points:
            subdiv.insert((int(additional_point[0]), int(additional_point[1])))

        triangle_list = subdiv.getTriangleList()
        triangle_polygons = triangle_list.reshape(-1, 3, 2)

        # ドロネー三角形用の色取得
        triangle_info_list = []
        for triangle_polygon in triangle_polygons:
            pt1 = (int(triangle_polygon[0][0]), int(triangle_polygon[0][1]))
            pt2 = (int(triangle_polygon[1][0]), int(triangle_polygon[1][1]))
            pt3 = (int(triangle_polygon[2][0]), int(triangle_polygon[2][1]))
            pt0 = (
                int((pt1[0] + pt2[0] + pt3[0]) / 3),
                int((pt1[1] + pt2[1] + pt3[1]) / 3),
            )
            color = tuple(image[pt0[1], pt0[0]])
            color = (int(color[0]), int(color[1]), int(color[2]))

            triangle_info_list.append([pt1, pt2, pt3, color])

        # 描画
        for triangle_info in triangle_info_list:
            pt1 = (int(triangle_info[0][0]), int(triangle_info[0][1]))
            pt2 = (int(triangle_info[1][0]), int(triangle_info[1][1]))
            pt3 = (int(triangle_info[2][0]), int(triangle_info[2][1]))
            contours = np.array([
                [pt1[0], pt1[1]],
                [pt2[0], pt2[1]],
                [pt3[0], pt3[1]],
            ])

            cv.fillConvexPoly(image, points=contours, color=triangle_info[3])

            if draw_line:
                cv.line(image, pt1, pt2, (255, 255, 255), 1, 8, 0)
                cv.line(image, pt2, pt3, (255, 255, 255), 1, 8, 0)
                cv.line(image, pt3, pt1, (255, 255, 255), 1, 8, 0)

        return image
