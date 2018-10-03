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

from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.parallel_interleave")
def parallel_interleave(map_func,
                        cycle_length,
                        block_length=1,
                        sloppy=False,
                        buffer_output_elements=None,
                        prefetch_input_elements=None):
  """A parallel version of the `Dataset.interleave()` transformation.

  `parallel_interleave()` maps `map_func` across its input to produce nested
  datasets, and outputs their elements interleaved. Unlike
  `tf.data.Dataset.interleave`, it gets elements from `cycle_length` nested
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
      tf.data.experimental.parallel_interleave(
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
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return readers.ParallelInterleaveDataset(
        dataset, map_func, cycle_length, block_length, sloppy,
        buffer_output_elements, prefetch_input_elements)

  return _apply_fn


class _DirectedInterleaveDataset(dataset_ops.Dataset):
  """A substitute for `Dataset.interleave()` on a fixed list of datasets."""

  def __init__(self, selector_input, data_inputs):
    self._selector_input = selector_input
    self._data_inputs = list(data_inputs)

    for data_input in data_inputs[1:]:
      if (data_input.output_types != data_inputs[0].output_types or
          data_input.output_classes != data_inputs[0].output_classes):
        raise TypeError("All datasets must have the same type and class.")

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return (
        gen_experimental_dataset_ops.experimental_directed_interleave_dataset(
            self._selector_input._as_variant_tensor(), [
                data_input._as_variant_tensor()
                for data_input in self._data_inputs
            ], **dataset_ops.flat_structure(self)))
    # pylint: enable=protected-access

  def _inputs(self):
    return [self._selector_input] + self._data_inputs

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


@tf_export("data.experimental.sample_from_datasets")
def sample_from_datasets(datasets, weights=None, seed=None):
  """Samples elements at random from the datasets in `datasets`.

  Args:
    datasets: A list of `tf.data.Dataset` objects with compatible structure.
    weights: (Optional.) A list of `len(datasets)` floating-point values where
      `weights[i]` represents the probability with which an element should be
      sampled from `datasets[i]`, or a `tf.data.Dataset` object where each
      element is such a list. Defaults to a uniform distribution across
      `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      random seed that will be used to create the distribution. See
      `tf.set_random_seed` for behavior.

  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.

  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError: If the `weights` argument is specified and does not match the
      length of the `datasets` element.
  """
  num_datasets = len(datasets)
  if not isinstance(weights, dataset_ops.Dataset):
    if weights is None:
      # Select inputs with uniform probability.
      logits = [[1.0] * num_datasets]

    else:
      # Use the given `weights` as the probability of choosing the respective
      # input.
      weights = ops.convert_to_tensor(weights, name="weights")
      if weights.dtype not in (dtypes.float32, dtypes.float64):
        raise TypeError("`weights` must be convertible to a tensor of "
                        "`tf.float32` or `tf.float64` elements.")
      if not weights.shape.is_compatible_with([num_datasets]):
        raise ValueError(
            "`weights` must be a vector of length `len(datasets)`.")

      # The `stateless_multinomial()` op expects log-probabilities, as opposed
      # to weights.
      logits = array_ops.expand_dims(math_ops.log(weights, name="logits"), 0)

    # NOTE(mrry): We only specialize when `weights` is not a `Dataset`. When it
    # is a `Dataset`, it is possible that evaluating it has a side effect the
    # user depends on.
    if len(datasets) == 1:
      return datasets[0]

    def select_dataset_constant_logits(seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed),
          axis=[0, 1])

    selector_input = dataset_ops.MapDataset(
        random_ops.RandomDataset(seed).batch(2),
        select_dataset_constant_logits,
        use_inter_op_parallelism=False)

  else:
    # Use each element of the given `weights` dataset as the probability of
    # choosing the respective input.

    # The `stateless_multinomial()` op expects log-probabilities, as opposed to
    # weights.
    logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))

    def select_dataset_varying_logits(logits, seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed),
          axis=[0, 1])

    logits_and_seeds = dataset_ops.Dataset.zip(
        (logits_ds, random_ops.RandomDataset(seed).batch(2)))
    selector_input = dataset_ops.MapDataset(
        logits_and_seeds,
        select_dataset_varying_logits,
        use_inter_op_parallelism=False)

  return _DirectedInterleaveDataset(selector_input, datasets)


@tf_export("data.experimental.choose_from_datasets")
def choose_from_datasets(datasets, choice_dataset):
  """Creates a dataset that deterministically chooses elements from `datasets`.

  For example, given the following datasets:

  ```python
  datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
              tf.data.Dataset.from_tensors("bar").repeat(),
              tf.data.Dataset.from_tensors("baz").repeat()]

  # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
  choice_dataset = tf.data.Dataset.range(3).repeat(3)

  result = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
  ```

  The elements of `result` will be:

  ```
  "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
  ```

  Args:
    datasets: A list of `tf.data.Dataset` objects with compatible structure.
    choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between
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
