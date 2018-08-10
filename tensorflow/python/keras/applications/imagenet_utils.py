# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import imagenet_utils
from tensorflow.python.util.tf_export import tf_export

decode_predictions = imagenet_utils.decode_predictions
preprocess_input = imagenet_utils.preprocess_input

tf_export(
    'keras.applications.imagenet_utils.decode_predictions',
    'keras.applications.densenet.decode_predictions',
    'keras.applications.inception_resnet_v2.decode_predictions',
    'keras.applications.inception_v3.decode_predictions',
    'keras.applications.mobilenet.decode_predictions',
    'keras.applications.mobilenet_v2.decode_predictions',
    'keras.applications.nasnet.decode_predictions',
    'keras.applications.resnet50.decode_predictions',
    'keras.applications.vgg16.decode_predictions',
    'keras.applications.vgg19.decode_predictions',
    'keras.applications.xception.decode_predictions',
)(decode_predictions)
tf_export(
    'keras.applications.imagenet_utils.preprocess_input',
    'keras.applications.resnet50.preprocess_input',
    'keras.applications.vgg16.preprocess_input',
    'keras.applications.vgg19.preprocess_input',
)(preprocess_input)
