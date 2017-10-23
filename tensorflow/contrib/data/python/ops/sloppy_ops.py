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
"""Non-deterministic dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


class SloppyInterleaveDataset(dataset_ops.Dataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length):
    """See `tf.contrib.data.sloppy_interleave()` for details."""
    super(SloppyInterleaveDataset, self).__init__()
    self._input_dataset = input_dataset

    @function.Defun(*nest.flatten(input_dataset.output_types))
    def tf_map_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      for arg, shape in zip(args, nest.flatten(input_dataset.output_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)

      if nest.is_sequence(nested_args):
        dataset = map_func(*nested_args)
      else:
        dataset = map_func(nested_args)

      if not isinstance(dataset, dataset_ops.Dataset):
        raise TypeError("`map_func` must return a `Dataset` object.")

      self._output_types = dataset.output_types
      self._output_shapes = dataset.output_shapes

      return dataset._as_variant_tensor()  # pylint: disable=protected-access

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")

  def _as_variant_tensor(self):
    return gen_dataset_ops.sloppy_interleave_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._map_func.captured_inputs,
        self._cycle_length,
        self._block_length,
        f=self._map_func,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


def sloppy_interleave(map_func, cycle_length, block_length=1):
  """A non-deterministic version of the `Dataset.interleave()` transformation.

  `sloppy_interleave()` maps `map_func` across `dataset`, and
  non-deterministically interleaves the results.

  The resulting dataset is almost identical to `interleave`. The key
  difference is that if retrieving a value from a given output iterator would
  cause `get_next` to block, that iterator will be skipped, and consumed
  when next available. If consuming from all iterators would cause the
  `get_next` call to block, the `get_next` call blocks until the first value is
  available.

  If the underlying datasets produce elements as fast as they are consumed, the
  `sloppy_interleave` transformation behaves identically to `interleave`.
  However, if an underlying dataset would block the consumer,
  `sloppy_interleave` can violate the round-robin order (that `interleave`
  strictly obeys), producing an element from a different underlying
  dataset instead.

  Example usage:

  ```python
  # Preprocess 4 files concurrently.
  filenames = tf.data.Dataset.list_files("/path/to/data/train*.tfrecords")
  dataset = filenames.apply(
      tf.contrib.data.sloppy_interleave(
          lambda filename: tf.data.TFRecordDataset(filename),
          cycle_length=4))
  ```

  WARNING: The order of elements in the resulting dataset is not
  deterministic. Use `Dataset.interleave()` if you want the elements to have a
  deterministic order.

  Args:
    map_func: A function mapping a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to a
      `Dataset`.
    cycle_length: The number of threads to interleave from in parallel.
    block_length: The number of consecutive elements to pull from a thread
      before advancing to the next thread. Note: sloppy_interleave will
      skip the remainder of elements in the block_length in order to avoid
      blocking.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return SloppyInterleaveDataset(
        dataset, map_func, cycle_length, block_length)
  return _apply_fn
