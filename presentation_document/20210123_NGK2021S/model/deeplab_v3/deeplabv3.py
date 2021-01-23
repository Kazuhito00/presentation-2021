#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv
import numpy as np
import tensorflow.compat.v1 as tf1


class DeepLabV3(object):
    def __init__(self):
        self.session = self._graph_load('model/deeplab_v3/deeplabv3_mnv2.pb')

    def __call__(self, image, inf_size=(480, 320)):
        INPUT_TENSOR_NAME = 'ImageTensor:0'
        OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

        temp_image = copy.deepcopy(image)
        temp_image = cv.resize(temp_image, inf_size)
        batch_segmentation_map = self.session.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(temp_image)]})
        segmentation_map = batch_segmentation_map[0]

        return segmentation_map

    def draw(self,
             image,
             segmentation_map,
             color=[0, 0, 0],
             inf_size=(480, 320)):
        image_width, image_height = image.shape[1], image.shape[0]

        draw_image = copy.deepcopy(image)
        draw_image = cv.resize(draw_image, inf_size)

        seg_image = self._label_to_color_image(segmentation_map,
                                               color).astype(np.uint8)
        seg_mask = self._label_to_person_mask(segmentation_map).astype(
            np.uint8)

        draw_image = np.where(seg_mask == 255, seg_image, draw_image)

        draw_image = cv.resize(draw_image, (image_width, image_height))

        return draw_image

    def transparent_draw(self,
                         image,
                         segmentation_map,
                         color=[0, 0, 0],
                         prev_frame=None,
                         inf_size=(480, 320)):
        image_width, image_height = image.shape[1], image.shape[0]

        draw_image = copy.deepcopy(image)
        if prev_frame is None:
            draw_image = cv.resize(draw_image, inf_size)
        else:
            draw_image = cv.resize(prev_frame, inf_size)

        seg_image = self._label_to_color_image(segmentation_map,
                                               color).astype(np.uint8)
        seg_mask = self._label_to_person_mask(segmentation_map).astype(
            np.uint8)

        draw_add_image = np.zeros((inf_size[1], inf_size[0], 3), np.uint8)
        draw_add_image = np.where(seg_mask == 255, seg_image, draw_add_image)

        draw_image = cv.add(draw_image, draw_add_image)

        draw_image = cv.resize(draw_image, (image_width, image_height))

        return draw_image

    def _graph_load(self, path):
        graph = tf1.Graph()
        graph_def = None
        with open(path, 'rb') as f:
            graph_def = tf1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf1.import_graph_def(graph_def, name='')
        config = tf1.ConfigProto(gpu_options=tf1.GPUOptions(allow_growth=True))
        sess = tf1.Session(graph=graph, config=config)
        return sess

    def _create_pascal_label_colormap(self, color):
        colormap = np.zeros((256, 3), dtype=int)

        ind = np.arange(256, dtype=int)
        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        colormap[15] = color  # 塗りつぶし色

        return colormap

    def _create_pascal_label_personmask(self):
        colormap = np.zeros((256, 3), dtype=int)
        colormap[15] = [255, 255, 255]
        return colormap

    def _label_to_color_image(self, label, color):
        colormap = self._create_pascal_label_colormap(color)
        return colormap[label]

    def _label_to_person_mask(self, label):
        colormap = self._create_pascal_label_personmask()
        return colormap[label]
