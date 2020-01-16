# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tensorflow V1 version of the Normalization preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine.base_preprocessing_layer_v1 import CombinerPreprocessingLayer
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.layers.experimental.preprocessing.Normalization'])
class Normalization(normalization.Normalization, CombinerPreprocessingLayer):
  pass
