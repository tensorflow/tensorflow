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
  """Wraps callable object as a `Layer` object.

  Args:
    func: The callable object to wrap.
    output_shape: A tuple with the (possibly partial) output shape of `func`
      *without* leading batch size (by analogy to Dense(..., input_shape=...)).
    trainable: Boolean controlling whether the trainable variables of `func`
      are reported as trainable variables of this layer.
  """

  def __init__(self, func, output_shape, trainable=False, **kwargs):
    # Set self._{non,}_trainable_weights before calling Layer.__init__.
    if hasattr(func, 'trainable_variables'):
      self._trainable_weights = [v for v in func.trainable_variables]
      trainable_variables_set = set(func.trainable_variables)
    else:
      self._trainable_weights = []
      trainable_variables_set = set()
    if hasattr(func, 'variables'):
      self._non_trainable_weights = [v for v in func.variables
                                     if v not in trainable_variables_set]
    else:
      self._non_trainable_weights = []  # TODO(arnoegw): Infer from `func`.
    super(CustomLayer, self).__init__(trainable=trainable, **kwargs)
    self._func = func
    # TODO(vbardiovsky): We should be able to get the embedding dimension from
    # the restored model.
    self._output_shape = tuple(output_shape)

  def call(self, x):
    result = self._func(x)
    # TODO(vbardiovsky): Polymorphic function should return shaped tensor.
    result.set_shape(self.compute_output_shape(x.shape))
    return result

  def compute_output_shape(self, input_shape):
    return (input_shape[0],) + self._output_shape
