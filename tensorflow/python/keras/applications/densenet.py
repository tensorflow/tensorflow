# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""DenseNet models for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import densenet

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.applications.densenet.DenseNet121',
              'keras.applications.DenseNet121')
@keras_modules_injection
def DenseNet121(*args, **kwargs):
  return densenet.DenseNet121(*args, **kwargs)


@keras_export('keras.applications.densenet.DenseNet169',
              'keras.applications.DenseNet169')
@keras_modules_injection
def DenseNet169(*args, **kwargs):
  return densenet.DenseNet169(*args, **kwargs)


@keras_export('keras.applications.densenet.DenseNet201',
              'keras.applications.DenseNet201')
@keras_modules_injection
def DenseNet201(*args, **kwargs):
  return densenet.DenseNet201(*args, **kwargs)


@keras_export('keras.applications.densenet.decode_predictions')
@keras_modules_injection
def decode_predictions(*args, **kwargs):
  return densenet.decode_predictions(*args, **kwargs)


@keras_export('keras.applications.densenet.preprocess_input')
@keras_modules_injection
def preprocess_input(*args, **kwargs):
  return densenet.preprocess_input(*args, **kwargs)
