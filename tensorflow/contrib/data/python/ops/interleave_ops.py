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

from tensorflow.contrib import stateless
from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.contrib.data.python.ops import random_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation


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
    return readers.ParallelInterleaveDataset(
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
    return readers.ParallelInterleaveDataset(
        dataset,
        map_func,
        cycle_length,
        block_length,
        sloppy=True,
        buffer_output_elements=None,
        prefetch_input_elements=None)

  return _apply_fn


class _DirectedInterleaveDataset(dataset_ops.Dataset):
  """A substitute for `Dataset.interleave()` on a fixed list of datasets."""

  def __init__(self, selector_input, data_inputs):
    self._selector_input = selector_input
    self._data_inputs = list(data_inputs)

    for data_input in data_inputs[1:]:
      if (data_input.output_types != data_inputs[0].output_types or
          data_input.output_classes != data_inputs[0].output_classes):
        raise TypeError("All datasets must have the same type.")

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.directed_interleave_dataset(
        self._selector_input._as_variant_tensor(),
        [data_input._as_variant_tensor() for data_input in self._data_inputs],
        **dataset_ops.flat_structure(self))
    # pylint: enable=protected-access

  @property
  def output_classes(self):
    return self._data_inputs[0].output_classes

  @property
  def output_shapes(self):
    ret = self._data_inputs[0].output_shapes
    for data_input in self._data_inputs[1:]:
      ret = nest.pack_sequence_as(ret, [
          ts1.most_specific_compatible_shape(ts2) for (ts1, ts2) in zip(
              nest.flatten(ret), nest.flatten(data_input.output_shapes))
      ])
    return ret

  @property
  def output_types(self):
    return self._data_inputs[0].output_types


def sample_from_datasets(datasets, weights=None, seed=None):
  """Samples elements at random from the datasets in `datasets`.

  Args:
    datasets: A list of @{tf.data.Dataset} objects with compatible structure.
    weights: (Optional.) A list of `len(datasets)` floating-point values where
      `weights[i]` represents the probability with which an element should be
      sampled from `datasets[i]`, or a @{tf.data.Dataset} object where each
      element is such a list. Defaults to a uniform distribution across
      `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      random seed that will be used to create the distribution. See
      @{tf.set_random_seed} for behavior.

  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.

  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError: If the `weights` argument is specified and does not match the
      length of the `datasets` element.
  """
  num_datasets = len(datasets)
  if weights is None:
    weights = dataset_ops.Dataset.from_tensors([1.0] * num_datasets).repeat()
  elif not isinstance(weights, dataset_ops.Dataset):
    weights = ops.convert_to_tensor(weights, name="weights")
    if weights.dtype not in (dtypes.float32, dtypes.float64):
      raise TypeError("`weights` must be convertible to a tensor of "
                      "`tf.float32` or `tf.float64` elements.")
    if not weights.shape.is_compatible_with([num_datasets]):
      raise ValueError("`weights` must be a vector of length `len(datasets)`.")
    weights = dataset_ops.Dataset.from_tensors(weights).repeat()

  # The `stateless_multinomial()` op expects log-probabilities, as opposed to
  # weights.
  logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))
  def select_dataset(logits, seed):
    return array_ops.squeeze(
        stateless.stateless_multinomial(logits, 1, seed=seed), axis=[0, 1])
  selector_input = dataset_ops.Dataset.zip(
      (logits_ds, random_ops.RandomDataset(seed).batch(2))).map(select_dataset)

  return _DirectedInterleaveDataset(selector_input, datasets)


def choose_from_datasets(datasets, choice_dataset):
  """Creates a dataset that deterministically chooses elements from `datasets`.

  For example, given the following datasets:

  ```python
  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
              tf.data.Dataset.from_tensors("bar").repeat(),
              tf.data.Dataset.from_tensors("baz").repeat()]

  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
  choice_dataset = tf.data.Dataset.range(3).repeat(3)

  result = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)
  ```

  The elements of `result` will be:

  ```
  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
  ```

  Args:
    datasets: A list of @{tf.data.Dataset} objects with compatible structure.
    choice_dataset: A @{tf.data.Dataset} of scalar `tf.int64` tensors between
      `0` and `len(datasets) - 1`.

  Returns:
    A dataset that interleaves elements from `datasets` according to the values
    of `choice_dataset`.

  Raises:
    TypeError: If the `datasets` or `choice_dataset` arguments have the wrong
      type.
  """
  if not (choice_dataset.output_types == dtypes.int64
          and choice_dataset.output_shapes.is_compatible_with(
              tensor_shape.scalar())
          and choice_dataset.output_classes == ops.Tensor):
    raise TypeError("`choice_dataset` must be a dataset of scalar "
                    "`tf.int64` tensors.")
  return _DirectedInterleaveDataset(choice_dataset, datasets)
