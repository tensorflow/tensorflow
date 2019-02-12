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
"""Utilities for integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# TODO(vbardiovsky): We should just reuse Keras's Lambda layer, when that
# enables to get trainable variables.
class CustomLayer(tf.keras.layers.Layer):
  """Wraps callable object as a `Layer` object."""

  def __init__(self, func, **kwargs):
    self._func = func
    super(CustomLayer, self).__init__(**kwargs)

  def call(self, x):
    result = self._func(x)
    # TODO(vbardiovsky): Polymorphic function should return shaped tensor.
    result.set_shape(self.compute_output_shape(x.shape))
    return result

  def compute_output_shape(self, input_shape):
    # TODO(vbardiovsky): We should be able to get the embedding dimension from
    # the restored model.
    return (input_shape[0], 10)

