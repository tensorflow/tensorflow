# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
"""Inception-ResNet V2 model for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import inception_resnet_v2

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.applications.inception_resnet_v2.InceptionResNetV2',
           'keras.applications.InceptionResNetV2')
@keras_modules_injection
def InceptionResNetV2(*args, **kwargs):
  return inception_resnet_v2.InceptionResNetV2(*args, **kwargs)


@tf_export('keras.applications.inception_resnet_v2.decode_predictions')
@keras_modules_injection
def decode_predictions(*args, **kwargs):
  return inception_resnet_v2.decode_predictions(*args, **kwargs)


@tf_export('keras.applications.inception_resnet_v2.preprocess_input')
@keras_modules_injection
def preprocess_input(*args, **kwargs):
  return inception_resnet_v2.preprocess_input(*args, **kwargs)
