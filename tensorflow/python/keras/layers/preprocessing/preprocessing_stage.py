# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Preprocessing stage."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.utils import tf_utils


class PreprocessingStage(base_preprocessing_layer.PreprocessingLayer,
                         sequential.Sequential):
  """A sequential preprocessing stage.

  This preprocessing stage wraps a list of preprocessing layers into a
  Sequential-like object that enables you to `adapt()` the whole list via
  a single `adapt()` call on the preprocessing stage.

  Arguments:
    layers: List of layers. Can include layers that aren't preprocessing layers.
    name: String. Optional name for the preprocessing stage object.
  """

  def adapt(self, data, reset_state=True):
    """Adapt the state of the layers of the preprocessing stage to the data.

    Arguments:
      data: A batched Dataset object, or a NumPy array, or an EagerTensor.
        Data to be iterated over to adapt the state of the layers in this
        preprocessing stage.
      reset_state: Whether this call to `adapt` should reset the state of
        the layers in this preprocessing stage.
    """
    if not isinstance(data,
                      (dataset_ops.DatasetV2, np.ndarray, ops.EagerTensor)):
      raise ValueError(
          '`adapt()` requires a batched Dataset, an EagerTensor, '
          'or a Numpy array as input, '
          'got {}'.format(type(data)))
    if isinstance(data, dataset_ops.DatasetV2):
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if tf_utils.dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to `adapt()` has an infinite number of '
            'elements. Please use dataset.take(...) to make the number '
            'of elements finite.')

    for current_layer_index in range(0, len(self.layers)):
      if not hasattr(self.layers[current_layer_index], 'adapt'):
        # Skip any layer that does not need adapting.
        continue

      def map_fn(x):
        """Maps `PreprocessingStage` inputs to inputs at `current_layer_index`.

        Args:
          x: Batch of inputs seen in entry of the `PreprocessingStage` instance.

        Returns:
          Batch of inputs to be processed by layer
            `self.layers[current_layer_index]`
        """
        if current_layer_index == 0:  # pylint: disable=cell-var-from-loop
          return x
        for i in range(current_layer_index):  # pylint: disable=cell-var-from-loop
          x = self.layers[i](x)
        return x

      if isinstance(data, dataset_ops.DatasetV2):
        current_layer_data = data.map(map_fn)
      else:
        current_layer_data = map_fn(data)
      self.layers[current_layer_index].adapt(current_layer_data,
                                             reset_state=reset_state)


