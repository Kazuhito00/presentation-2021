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
                'pose_smooth_landmarks': True,
                'pose_min_detection_confidence': 0.5,
                'pose_min_tracking_confidence': 0.5,
                'pose_model_complexity': 1,
                'holistic_static_image_mode': False,
                'holistic_smooth_landmarks': True,
                'holistic_min_detection_confidence': 0.5,
                'holistic_min_tracking_confidence': 0.5,
                'face_detection_min_detection_confidence': 0.7,
                'objectron_static_image_mode': False,
                'objectron_max_num_objects': 5,
                'objectron_min_detection_confidence': 0.5,
                'objectron_min_tracking_confidence': 0.99,
                'objectron_model_name': 'Shoe',
            }
            # objectron_model_name{'Shoe', 'Chair', 'Cup', 'Camera'}
        else:
            self._config = config

        self.hand = self._create_hands_solution()
        self.face_mesh = self._create_face_mesh_solution()
        self.pose = self._create_pose_solution()
        self.holistic = self._create_holistic_solution()
        self.face_detection = self._create_face_detection_solution()
        self.objectron = self._create_objectron_solution()

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

    def _create_holistic_solution(self):
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=self._config['holistic_static_image_mode'],
            smooth_landmarks=self._config['holistic_smooth_landmarks'],
            min_detection_confidence=self.
            _config['holistic_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['holistic_min_tracking_confidence'],
        )
        return holistic

    def _create_face_detection_solution(self):
        face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.
            _config['face_detection_min_detection_confidence'])
        return face_detection

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
            smooth_landmarks=self._config['pose_smooth_landmarks'],
            min_detection_confidence=self.
            _config['pose_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['pose_min_tracking_confidence'],
            model_complexity=self._config['pose_model_complexity'],
        )
        return pose

    def _create_objectron_solution(self):
        self._mp_drawing = mp.solutions.drawing_utils

        self._objectron_solution = mp.solutions.objectron
        objectron = mp.solutions.objectron.Objectron(
            static_image_mode=self._config['objectron_static_image_mode'],
            max_num_objects=self._config['objectron_max_num_objects'],
            min_detection_confidence=self.
            _config['objectron_min_detection_confidence'],
            min_tracking_confidence=self.
            _config['objectron_min_tracking_confidence'],
            model_name=self._config['objectron_model_name'],
        )
        return objectron

    def __call__(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        hand_results = self.hand.process(rgb_image)
        face_mesh_results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        return face_mesh_results, hand_results

    def process_hands(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        hand_results = self.hand.process(rgb_image)
        rgb_image.flags.writeable = True

        return hand_results

    def process_face_mesh(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        face_mesh_results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        return face_mesh_results

    def process_pose(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        pose_results = self.pose.process(rgb_image)
        rgb_image.flags.writeable = True

        return pose_results

    def process_holistic(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        holistic_results = self.holistic.process(rgb_image)
        rgb_image.flags.writeable = True

        return holistic_results

    def process_face_detection(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        face_detection_results = self.face_detection.process(rgb_image)
        rgb_image.flags.writeable = True

        return face_detection_results

    def process_objectron(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        rgb_image.flags.writeable = False
        objectron_results = self.objectron.process(rgb_image)
        rgb_image.flags.writeable = True

        return objectron_results

    def draw01(self, image, face_detection_results):
        if face_detection_results.detections is not None:
            for detection in face_detection_results.detections:
                image = self._draw_face_detection(image, detection)
        return image

    def draw02(self, image, face_mesh_results):
        if face_mesh_results.multi_face_landmarks is not None:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # 外接矩形の計算
                brect = self._calc_bounding_rect(image, face_landmarks)
                # 描画
                image = self._draw_face_mesh(image, face_landmarks)
                image = self._draw_bounding_rect(image, brect)
        return image

    def draw03(self, image, pose_results):
        # 描画
        if pose_results is not None:
            if pose_results.pose_landmarks is not None:
                image = self._draw_pose(image, pose_results.pose_landmarks)
        return image

    def draw04(self, image, hands_results):
        if hands_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    hands_results.multi_hand_landmarks,
                    hands_results.multi_handedness):
                # 手の平重心計算
                cx, cy = self._calc_palm_moment(image, hand_landmarks)
                # 描画
                image = self._draw_hands(image, cx, cy, hand_landmarks,
                                         handedness.classification[0].label[0])
        return image

    def draw05(self, image, holistic_results):
        if holistic_results is None:
            return image

        # Face Mesh
        face_landmarks = holistic_results.face_landmarks
        if face_landmarks is not None:
            # 外接矩形の計算
            brect = self._calc_bounding_rect(image, face_landmarks)
            # 描画
            image = self._draw_face_mesh(image, face_landmarks)
            image = self._draw_bounding_rect(image, brect)

        # Pose
        pose_landmarks = holistic_results.pose_landmarks
        if pose_landmarks is not None:
            image = self._draw_pose(image, pose_landmarks)

        # Hands
        left_hand_landmarks = holistic_results.left_hand_landmarks
        right_hand_landmarks = holistic_results.right_hand_landmarks
        # 左手
        if left_hand_landmarks is not None:
            # 手の平重心計算
            cx, cy = self._calc_palm_moment(image, left_hand_landmarks)
            # 描画
            image = self._draw_hands(image, cx, cy, left_hand_landmarks, 'R')

        # 右手
        if right_hand_landmarks is not None:
            # 手の平重心計算
            cx, cy = self._calc_palm_moment(image, right_hand_landmarks)
            # 描画
            image = self._draw_hands(image, cx, cy, right_hand_landmarks, 'L')

        return image

    def draw06(self, image, objectron_results):
        if objectron_results.detected_objects is not None:
            for detected_object in objectron_results.detected_objects:
                self._mp_drawing.draw_landmarks(
                    image, detected_object.landmarks_2d,
                    self._objectron_solution.BOX_CONNECTIONS)
                self._mp_drawing.draw_axis(image, detected_object.rotation,
                                           detected_object.translation)

                # キーポイント確認用
                self._draw_objectron(image, detected_object.landmarks_2d)
        return image

    def draw11(self, image, face_results, hand_results):
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

    def _draw_face_detection(self, image, process_result):
        image_width, image_height = image.shape[1], image.shape[0]

        # バウンディングボックス
        bbox = process_result.location_data.relative_bounding_box
        bbox.xmin = int(bbox.xmin * image_width)
        bbox.ymin = int(bbox.ymin * image_height)
        bbox.width = int(bbox.width * image_width)
        bbox.height = int(bbox.height * image_height)

        cv.rectangle(
            image, (int(bbox.xmin), int(bbox.ymin)),
            (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
            (0, 255, 0), 2)

        # スコア・ラベルID
        # cv.putText(
        #     image,
        #     str(process_result.label_id[0]) + ":" +
        #     str(round(process_result.score[0], 3)),
        #     (int(bbox.xmin), int(bbox.ymin) - 20), cv.FONT_HERSHEY_SIMPLEX,
        #     1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キーポイント：右目
        keypoint0 = process_result.location_data.relative_keypoints[0]
        keypoint0.x = int(keypoint0.x * image_width)
        keypoint0.y = int(keypoint0.y * image_height)

        cv.circle(image, (int(keypoint0.x), int(keypoint0.y)), 5, (0, 255, 0),
                  2)

        # キーポイント：左目
        keypoint1 = process_result.location_data.relative_keypoints[1]
        keypoint1.x = int(keypoint1.x * image_width)
        keypoint1.y = int(keypoint1.y * image_height)

        cv.circle(image, (int(keypoint1.x), int(keypoint1.y)), 5, (0, 255, 0),
                  2)

        # キーポイント：鼻
        keypoint2 = process_result.location_data.relative_keypoints[2]
        keypoint2.x = int(keypoint2.x * image_width)
        keypoint2.y = int(keypoint2.y * image_height)

        cv.circle(image, (int(keypoint2.x), int(keypoint2.y)), 5, (0, 255, 0),
                  2)

        # キーポイント：口
        keypoint3 = process_result.location_data.relative_keypoints[3]
        keypoint3.x = int(keypoint3.x * image_width)
        keypoint3.y = int(keypoint3.y * image_height)

        cv.circle(image, (int(keypoint3.x), int(keypoint3.y)), 5, (0, 255, 0),
                  2)

        # キーポイント：右耳
        keypoint4 = process_result.location_data.relative_keypoints[4]
        keypoint4.x = int(keypoint4.x * image_width)
        keypoint4.y = int(keypoint4.y * image_height)

        cv.circle(image, (int(keypoint4.x), int(keypoint4.y)), 5, (0, 255, 0),
                  2)

        # キーポイント：左耳
        keypoint5 = process_result.location_data.relative_keypoints[5]
        keypoint5.x = int(keypoint5.x * image_width)
        keypoint5.y = int(keypoint5.y * image_height)

        cv.circle(image, (int(keypoint5.x), int(keypoint5.y)), 5, (0, 255, 0),
                  2)

        return image

    def _draw_face_mesh(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append((landmark_x, landmark_y))

            cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

        if len(landmark_point) > 0:
            # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

            # 左眉毛(55：内側、46：外側)
            cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0),
                    2)

            # 右眉毛(285：内側、276：外側)
            cv.line(image, landmark_point[285], landmark_point[295],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[295], landmark_point[282],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[282], landmark_point[283],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[283], landmark_point[276],
                    (0, 255, 0), 2)

            # 左目 (133：目頭、246：目尻)
            cv.line(image, landmark_point[133], landmark_point[173],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[173], landmark_point[157],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[157], landmark_point[158],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[158], landmark_point[159],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[159], landmark_point[160],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[160], landmark_point[161],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[161], landmark_point[246],
                    (0, 255, 0), 2)

            cv.line(image, landmark_point[246], landmark_point[163],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[163], landmark_point[144],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[144], landmark_point[145],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[145], landmark_point[153],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[153], landmark_point[154],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[154], landmark_point[155],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[155], landmark_point[133],
                    (0, 255, 0), 2)

            # 右目 (362：目頭、466：目尻)
            cv.line(image, landmark_point[362], landmark_point[398],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[398], landmark_point[384],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[384], landmark_point[385],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[385], landmark_point[386],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[386], landmark_point[387],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[387], landmark_point[388],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[388], landmark_point[466],
                    (0, 255, 0), 2)

            cv.line(image, landmark_point[466], landmark_point[390],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[390], landmark_point[373],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[373], landmark_point[374],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[374], landmark_point[380],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[380], landmark_point[381],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[381], landmark_point[382],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[382], landmark_point[362],
                    (0, 255, 0), 2)

            # 口 (308：右端、78：左端)
            cv.line(image, landmark_point[308], landmark_point[415],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[415], landmark_point[310],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[310], landmark_point[311],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[311], landmark_point[312],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[312], landmark_point[13],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[80], landmark_point[191],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[191], landmark_point[78],
                    (0, 255, 0), 2)

            cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[88], landmark_point[178],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[178], landmark_point[87],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[14], landmark_point[317],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[317], landmark_point[402],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[402], landmark_point[318],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[318], landmark_point[324],
                    (0, 255, 0), 2)
            cv.line(image, landmark_point[324], landmark_point[308],
                    (0, 255, 0), 2)

        return image

    def _draw_pose(self, image, landmarks, visibility_th=0.5):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append(
                [landmark.visibility, (landmark_x, landmark_y)])

            if landmark.visibility < visibility_th:
                continue

            if index == 0:  # 鼻
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # 右目：目頭
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # 右目：瞳
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # 右目：目尻
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # 左目：目頭
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 5:  # 左目：瞳
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # 左目：目尻
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # 右耳
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # 左耳
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 9:  # 口：左端
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # 口：左端
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # 右肩
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # 左肩
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 13:  # 右肘
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # 左肘
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # 右手首
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # 左手首
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 17:  # 右手1(外側端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # 左手1(外側端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # 右手2(先端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # 左手2(先端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 21:  # 右手3(内側端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 22:  # 左手3(内側端)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 23:  # 腰(右側)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 24:  # 腰(左側)
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 25:  # 右ひざ
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 26:  # 左ひざ
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 27:  # 右足首
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 28:  # 左足首
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 29:  # 右かかと
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 30:  # 左かかと
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 31:  # 右つま先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 32:  # 左つま先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

            # if not upper_body_only:
            if True:
                cv.putText(image, "z:" + str(round(landmark_z, 3)),
                           (landmark_x - 10, landmark_y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                           cv.LINE_AA)

        if len(landmark_point) > 0:
            # 右目
            if landmark_point[1][0] > visibility_th and landmark_point[2][
                    0] > visibility_th:
                cv.line(image, landmark_point[1][1], landmark_point[2][1],
                        (0, 255, 0), 2)
            if landmark_point[2][0] > visibility_th and landmark_point[3][
                    0] > visibility_th:
                cv.line(image, landmark_point[2][1], landmark_point[3][1],
                        (0, 255, 0), 2)

            # 左目
            if landmark_point[4][0] > visibility_th and landmark_point[5][
                    0] > visibility_th:
                cv.line(image, landmark_point[4][1], landmark_point[5][1],
                        (0, 255, 0), 2)
            if landmark_point[5][0] > visibility_th and landmark_point[6][
                    0] > visibility_th:
                cv.line(image, landmark_point[5][1], landmark_point[6][1],
                        (0, 255, 0), 2)

            # 口
            if landmark_point[9][0] > visibility_th and landmark_point[10][
                    0] > visibility_th:
                cv.line(image, landmark_point[9][1], landmark_point[10][1],
                        (0, 255, 0), 2)

            # 肩
            if landmark_point[11][0] > visibility_th and landmark_point[12][
                    0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[12][1],
                        (0, 255, 0), 2)

            # 右腕
            if landmark_point[11][0] > visibility_th and landmark_point[13][
                    0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[13][1],
                        (0, 255, 0), 2)
            if landmark_point[13][0] > visibility_th and landmark_point[15][
                    0] > visibility_th:
                cv.line(image, landmark_point[13][1], landmark_point[15][1],
                        (0, 255, 0), 2)

            # 左腕
            if landmark_point[12][0] > visibility_th and landmark_point[14][
                    0] > visibility_th:
                cv.line(image, landmark_point[12][1], landmark_point[14][1],
                        (0, 255, 0), 2)
            if landmark_point[14][0] > visibility_th and landmark_point[16][
                    0] > visibility_th:
                cv.line(image, landmark_point[14][1], landmark_point[16][1],
                        (0, 255, 0), 2)

            # 右手
            if landmark_point[15][0] > visibility_th and landmark_point[17][
                    0] > visibility_th:
                cv.line(image, landmark_point[15][1], landmark_point[17][1],
                        (0, 255, 0), 2)
            if landmark_point[17][0] > visibility_th and landmark_point[19][
                    0] > visibility_th:
                cv.line(image, landmark_point[17][1], landmark_point[19][1],
                        (0, 255, 0), 2)
            if landmark_point[19][0] > visibility_th and landmark_point[21][
                    0] > visibility_th:
                cv.line(image, landmark_point[19][1], landmark_point[21][1],
                        (0, 255, 0), 2)
            if landmark_point[21][0] > visibility_th and landmark_point[15][
                    0] > visibility_th:
                cv.line(image, landmark_point[21][1], landmark_point[15][1],
                        (0, 255, 0), 2)

            # 左手
            if landmark_point[16][0] > visibility_th and landmark_point[18][
                    0] > visibility_th:
                cv.line(image, landmark_point[16][1], landmark_point[18][1],
                        (0, 255, 0), 2)
            if landmark_point[18][0] > visibility_th and landmark_point[20][
                    0] > visibility_th:
                cv.line(image, landmark_point[18][1], landmark_point[20][1],
                        (0, 255, 0), 2)
            if landmark_point[20][0] > visibility_th and landmark_point[22][
                    0] > visibility_th:
                cv.line(image, landmark_point[20][1], landmark_point[22][1],
                        (0, 255, 0), 2)
            if landmark_point[22][0] > visibility_th and landmark_point[16][
                    0] > visibility_th:
                cv.line(image, landmark_point[22][1], landmark_point[16][1],
                        (0, 255, 0), 2)

            # 胴体
            if landmark_point[11][0] > visibility_th and landmark_point[23][
                    0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[23][1],
                        (0, 255, 0), 2)
            if landmark_point[12][0] > visibility_th and landmark_point[24][
                    0] > visibility_th:
                cv.line(image, landmark_point[12][1], landmark_point[24][1],
                        (0, 255, 0), 2)
            if landmark_point[23][0] > visibility_th and landmark_point[24][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[24][1],
                        (0, 255, 0), 2)

            if len(landmark_point) > 25:
                # 右足
                if landmark_point[23][0] > visibility_th and landmark_point[
                        25][0] > visibility_th:
                    cv.line(image, landmark_point[23][1],
                            landmark_point[25][1], (0, 255, 0), 2)
                if landmark_point[25][0] > visibility_th and landmark_point[
                        27][0] > visibility_th:
                    cv.line(image, landmark_point[25][1],
                            landmark_point[27][1], (0, 255, 0), 2)
                if landmark_point[27][0] > visibility_th and landmark_point[
                        29][0] > visibility_th:
                    cv.line(image, landmark_point[27][1],
                            landmark_point[29][1], (0, 255, 0), 2)
                if landmark_point[29][0] > visibility_th and landmark_point[
                        31][0] > visibility_th:
                    cv.line(image, landmark_point[29][1],
                            landmark_point[31][1], (0, 255, 0), 2)

                # 左足
                if landmark_point[24][0] > visibility_th and landmark_point[
                        26][0] > visibility_th:
                    cv.line(image, landmark_point[24][1],
                            landmark_point[26][1], (0, 255, 0), 2)
                if landmark_point[26][0] > visibility_th and landmark_point[
                        28][0] > visibility_th:
                    cv.line(image, landmark_point[26][1],
                            landmark_point[28][1], (0, 255, 0), 2)
                if landmark_point[28][0] > visibility_th and landmark_point[
                        30][0] > visibility_th:
                    cv.line(image, landmark_point[28][1],
                            landmark_point[30][1], (0, 255, 0), 2)
                if landmark_point[30][0] > visibility_th and landmark_point[
                        32][0] > visibility_th:
                    cv.line(image, landmark_point[30][1],
                            landmark_point[32][1], (0, 255, 0), 2)
        return image

    def _draw_hands(self, image, cx, cy, landmarks, handedness_str):
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
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # 手首2
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0),
                    2)

            # 人差指
            cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0),
                    2)

            # 中指
            cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0),
                    2)

            # 薬指
            cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0),
                    2)

            # 小指
            cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0),
                    2)

            # 手の平
            cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0),
                    2)
            cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0),
                    2)

        # 重心 + 左右
        if len(landmark_point) > 0:
            # cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
            cv.putText(image, handedness_str, (cx - 6, cy + 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                       cv.LINE_AA)

        return image

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

    def _draw_objectron(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([(landmark_x, landmark_y)])

            if index == 0:  # 重心
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 5:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  #
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        return image

    def _draw_bounding_rect(self, image, brect):
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)
        return image

    def _calc_palm_moment(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        palm_array = np.empty((0, 2), int)

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            if index == 0:  # 手首1
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # 手首2
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # 人差指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # 中指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # 薬指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # 小指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
        M = cv.moments(palm_array)
        cx, cy = 0, 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        return cx, cy
