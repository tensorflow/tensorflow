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
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import deprecation


class ParallelInterleaveDataset(dataset_ops.Dataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length,
               sloppy, buffer_output_elements, prefetch_input_elements):
    """See `tf.contrib.data.parallel_interleave()` for details."""
    super(ParallelInterleaveDataset, self).__init__()
    self._input_dataset = input_dataset

    @function.Defun(*nest.flatten(
        sparse.as_dense_types(input_dataset.output_types,
                              input_dataset.output_classes)))
    def tf_map_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      dense_shapes = sparse.as_dense_shapes(input_dataset.output_shapes,
                                            input_dataset.output_classes)
      for arg, shape in zip(args, nest.flatten(dense_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)
      nested_args = sparse.deserialize_sparse_tensors(
          nested_args, input_dataset.output_types, input_dataset.output_shapes,
          input_dataset.output_classes)
      if dataset_ops._should_unpack_args(nested_args):  # pylint: disable=protected-access
        dataset = map_func(*nested_args)
      else:
        dataset = map_func(nested_args)

      if not isinstance(dataset, dataset_ops.Dataset):
        raise TypeError("`map_func` must return a `Dataset` object.")

      self._output_classes = dataset.output_classes
      self._output_types = dataset.output_types
      self._output_shapes = dataset.output_shapes

      return dataset._as_variant_tensor()  # pylint: disable=protected-access

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._sloppy = ops.convert_to_tensor(
        sloppy, dtype=dtypes.bool, name="sloppy")
    self._buffer_output_elements = convert.optional_param_to_tensor(
        "buffer_output_elements",
        buffer_output_elements,
        argument_default=2 * block_length)
    self._prefetch_input_elements = convert.optional_param_to_tensor(
        "prefetch_input_elements",
        prefetch_input_elements,
        argument_default=2 * cycle_length)

  def _as_variant_tensor(self):
    return gen_dataset_ops.parallel_interleave_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._map_func.captured_inputs,
        self._cycle_length,
        self._block_length,
        self._sloppy,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        f=self._map_func,
        output_types=nest.flatten(
            sparse.as_dense_types(self.output_types, self.output_classes)),
        output_shapes=nest.flatten(
            sparse.as_dense_shapes(self.output_shapes, self.output_classes)))

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


def parallel_interleave(map_func,
                        cycle_length,
                        block_length=1,
                        sloppy=False,
                        buffer_output_elements=None,
                        prefetch_input_elements=None):
  """A parallel version of the `Dataset.interleave()` transformation.

  `parallel_interleave()` maps `map_func` across its input to produce nested
  datasets, and outputs their elements interleaved. Unlike
  @{tf.data.Dataset.interleave}, it gets elements from `cycle_length` nested
  datasets in parallel, which increases the throughput, especially in the
  presence of stragglers. Furthermore, the `sloppy` argument can be used to
  improve performance, by relaxing the requirement that the outputs are produced
  in a deterministic order, and allowing the implementation to skip over nested
  datasets whose elements are not readily available when requested.

  Example usage:

  ```python
  # Preprocess 4 files concurrently.
  filenames = tf.data.Dataset.list_files("/path/to/data/train*.tfrecords")
  dataset = filenames.apply(
      tf.contrib.data.parallel_interleave(
          lambda filename: tf.data.TFRecordDataset(filename),
          cycle_length=4))
  ```

  WARNING: If `sloppy` is `True`, the order of produced elements is not
  deterministic.

  Args:
    map_func: A function mapping a nested structure of tensors to a `Dataset`.
    cycle_length: The number of input `Dataset`s to interleave from in parallel.
    block_length: The number of consecutive elements to pull from an input
      `Dataset` before advancing to the next input `Dataset`.
    sloppy: If false, elements are produced in deterministic order. Otherwise,
      the implementation is allowed, for the sake of expediency, to produce
      elements in a non-deterministic order.
    buffer_output_elements: The number of elements each iterator being
      interleaved should buffer (similar to the `.prefetch()` transformation for
      each interleaved iterator).
    prefetch_input_elements: The number of input elements to transform to
      iterators before they are needed for interleaving.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return ParallelInterleaveDataset(
        dataset, map_func, cycle_length, block_length, sloppy,
        buffer_output_elements, prefetch_input_elements)

  return _apply_fn


@deprecation.deprecated(
    None, "Use `tf.contrib.data.parallel_interleave(..., sloppy=True)`.")
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
    cycle_length: The number of input `Dataset`s to interleave from in parallel.
    block_length: The number of consecutive elements to pull from an input
      `Dataset` before advancing to the next input `Dataset`. Note:
      `sloppy_interleave` will skip the remainder of elements in the
      `block_length` in order to avoid blocking.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return ParallelInterleaveDataset(
        dataset,
        map_func,
        cycle_length,
        block_length,
        sloppy=True,
        buffer_output_elements=None,
        prefetch_input_elements=None)

  return _apply_fn
