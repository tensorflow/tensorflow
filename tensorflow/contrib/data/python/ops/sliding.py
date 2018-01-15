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
"""Batching dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops


class SlideDataset(dataset_ops.Dataset):
  """A `Dataset` that slides a fixed window on its input."""

  def __init__(self, input_dataset, slide_size, slide_step=1):
    """See `Dataset.slide()` for details."""
    super(SlideDataset, self).__init__()
    self._input_dataset = input_dataset
    self._slide_size = ops.convert_to_tensor(
      slide_size, dtype=dtypes.int64, name="slide_size")
    self._slide_step = ops.convert_to_tensor(
      slide_step, dtype=dtypes.int64, name="slide_step")

  def _as_variant_tensor(self):
    return gen_dataset_ops._slide_dataset(  # pylint: disable=protected-access
      self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
      slide_size=self._slide_size,
      slide_step=self._slide_step,
      output_shapes=nest.flatten(
        sparse.as_dense_shapes(self.output_shapes, self.output_classes)),
      output_types=nest.flatten(
        sparse.as_dense_types(self.output_types, self.output_classes)))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    input_shapes = self._input_dataset.output_shapes
    return nest.pack_sequence_as(input_shapes, [
      tensor_shape.vector(None).concatenate(s)
      for s in nest.flatten(self._input_dataset.output_shapes)
    ])

  @property
  def output_types(self):
    return self._input_dataset.output_types
