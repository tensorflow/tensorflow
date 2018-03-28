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
"""Sliding dataset transformations."""
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


class _SlideDataset(dataset_ops.Dataset):
  """A `Dataset` that passes a sliding window over its input."""

  def __init__(self, input_dataset, window_size, stride=1):
    """See `sliding_window_batch` for details."""
    super(_SlideDataset, self).__init__()
    self._input_dataset = input_dataset
    self._window_size = ops.convert_to_tensor(
        window_size, dtype=dtypes.int64, name="window_size")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")

  def _as_variant_tensor(self):
    return gen_dataset_ops.slide_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        window_size=self._window_size,
        stride=self._stride,
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


def sliding_window_batch(window_size, stride=1):
  """A sliding window with size of `window_size` and step of `stride`.

  This transformation passes a sliding window over this dataset. The
  window size is `window_size` and step size is `stride`. If the left
  elements cannot fill up the sliding window, this transformation will
  drop the final smaller element. For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { [1], [2], [3], [4], [5], [6] }

  a.apply(tf.contrib.data.sliding_window_batch(window_size=3, stride=2)) ==
  {
      [[1], [2], [3]],
      [[3], [4], [5]],
  }
  ```

  Args:
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      elements in the sliding window.
    stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      steps moving the sliding window forward for one iteration. The default
      is `1`. It must be in `[1, window_size)`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return _SlideDataset(dataset, window_size, stride)

  return _apply_fn
