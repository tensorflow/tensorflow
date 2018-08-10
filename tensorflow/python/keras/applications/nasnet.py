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
"""NASNet-A models for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import nasnet
from tensorflow.python.util.tf_export import tf_export

NASNetMobile = nasnet.NASNetMobile
NASNetLarge = nasnet.NASNetLarge
decode_predictions = nasnet.decode_predictions
preprocess_input = nasnet.preprocess_input

tf_export('keras.applications.nasnet.NASNetMobile',
          'keras.applications.NASNetMobile')(NASNetMobile)
tf_export('keras.applications.nasnet.NASNetLarge',
          'keras.applications.NASNetLarge')(NASNetLarge)
tf_export('keras.applications.nasnet.preprocess_input')(preprocess_input)
