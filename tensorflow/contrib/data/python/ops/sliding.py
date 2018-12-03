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
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation


class _SlideDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that passes a sliding window over its input."""

  def __init__(self, input_dataset, window_size, window_shift, window_stride):
    """See `sliding_window_batch` for details."""
    super(_SlideDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._window_size = ops.convert_to_tensor(
        window_size, dtype=dtypes.int64, name="window_stride")
    self._window_stride = ops.convert_to_tensor(
        window_stride, dtype=dtypes.int64, name="window_stride")
    self._window_shift = ops.convert_to_tensor(
        window_shift, dtype=dtypes.int64, name="window_shift")

    # pylint: disable=protected-access
    input_structure = structure.Structure._from_legacy_structure(
        input_dataset.output_types, input_dataset.output_shapes,
        input_dataset.output_classes)
    self._output_structure = input_structure._batch(None)

  def _as_variant_tensor(self):
    return ged_ops.experimental_sliding_window_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        window_size=self._window_size,
        window_shift=self._window_shift,
        window_stride=self._window_stride,
        **dataset_ops.flat_structure(structure=self._output_structure))

  @property
  def output_classes(self):
    return self._output_structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._output_structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._output_structure._to_legacy_output_types()  # pylint: disable=protected-access


@deprecation.deprecated_args(
    None, "stride is deprecated, use window_shift instead", "stride")
@deprecation.deprecated(
    None, "Use `tf.data.Dataset.window(size=window_size, shift=window_shift, "
    "stride=window_stride).flat_map(lambda x: x.batch(window.size))` "
    "instead.")
def sliding_window_batch(window_size,
                         stride=None,
                         window_shift=None,
                         window_stride=1):
  """A sliding window over a dataset.

  This transformation passes a sliding window over this dataset. The window size
  is `window_size`, the stride of the input elements is `window_stride`, and the
  shift between consecutive windows is `window_shift`. If the remaining elements
  cannot fill up the sliding window, this transformation will drop the final
  smaller element. For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { [1], [2], [3], [4], [5], [6] }

  a.apply(sliding_window_batch(window_size=3)) ==
  { [[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]] }

  a.apply(sliding_window_batch(window_size=3, window_shift=2)) ==
  { [[1], [2], [3]], [[3], [4], [5]] }

  a.apply(sliding_window_batch(window_size=3, window_stride=2)) ==
  { [[1], [3], [5]], [[2], [4], [6]] }
  ```

  Args:
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      elements in the sliding window. It must be positive.
    stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      forward shift of the sliding window in each iteration. The default is `1`.
      It must be positive. Deprecated alias for `window_shift`.
    window_shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      forward shift of the sliding window in each iteration. The default is `1`.
      It must be positive.
    window_stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      stride of the input elements in the sliding window. The default is `1`.
      It must be positive.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if invalid arguments are provided.
  """
  if stride is None and window_shift is None:
    window_shift = 1
  elif stride is not None and window_shift is None:
    window_shift = stride
  elif stride is not None and window_shift is not None:
    raise ValueError("Cannot specify both `stride` and `window_shift`")

  def _apply_fn(dataset):
    return _SlideDataset(dataset, window_size, window_shift, window_stride)

  return _apply_fn
