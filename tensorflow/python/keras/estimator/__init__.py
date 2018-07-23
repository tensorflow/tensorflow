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
"""Keras estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export

# Keras has undeclared dependency on tensorflow/estimator:estimator_py.
# As long as you depend //third_party/py/tensorflow:tensorflow target
# everything will work as normal.

try:
  from tensorflow.python.estimator import keras as keras_lib  # pylint: disable=g-import-not-at-top
  model_to_estimator = tf_export('keras.estimator.model_to_estimator')(
      keras_lib.model_to_estimator)
except Exception:  # pylint: disable=broad-except

  # pylint: disable=unused-argument
  def stub_model_to_estimator(keras_model=None,
                              keras_model_path=None,
                              custom_objects=None,
                              model_dir=None,
                              config=None):
    raise NotImplementedError(
        'tf.keras.estimator.model_to_estimator function not available in your '
        'installation.')
  # pylint: enable=unused-argument

  model_to_estimator = tf_export('keras.estimator.model_to_estimator')(
      stub_model_to_estimator)

