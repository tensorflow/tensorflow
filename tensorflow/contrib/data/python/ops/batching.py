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

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import gfile


def dense_to_sparse_batch(batch_size, row_shape):
  """A transformation that batches ragged elements into `tf.SparseTensor`s.

  Like `Dataset.padded_batch()`, this transformation combines multiple
  consecutive elements of the dataset, which might have different
  shapes, into a single element. The resulting element has three
  components (`indices`, `values`, and `dense_shape`), which
  comprise a `tf.SparseTensor` that represents the same data. The
  `row_shape` represents the dense shape of each row in the
  resulting `tf.SparseTensor`, to which the effective batch size is
  prepended. For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

  a.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=2, row_shape=[6])) ==
  {
      ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
       ['a', 'b', 'c', 'a', 'b'],                 # values
       [2, 6]),                                   # dense_shape
      ([[2, 0], [2, 1], [2, 2], [2, 3]],
       ['a', 'b', 'c', 'd'],
       [1, 6])
  }
  ```

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the
      number of consecutive elements of this dataset to combine in a
      single batch.
    row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like
      object representing the equivalent dense shape of a row in the
      resulting `tf.SparseTensor`. Each element of this dataset must
      have the same rank as `row_shape`, and must have size less
      than or equal to `row_shape` in each dimension.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return DenseToSparseBatchDataset(dataset, batch_size, row_shape)

  return _apply_fn


def unbatch():
  """A Transformation which splits the elements of a dataset.

  For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
  where `B` may vary from element to element, then for each element in
  the dataset, the unbatched dataset will contain `B` consecutive elements
  of shape `[a0, a1, ...]`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):

    def unbatch_map(arg, *rest):
      if rest:
        return dataset_ops.Dataset.from_tensor_slices((arg,) + rest)
      else:
        return dataset_ops.Dataset.from_tensor_slices(arg)

    return dataset.flat_map(map_func=unbatch_map)

  return _apply_fn


def _calculate_acceptance_probs(initial_probs, target_probs):
  """Calculate the per-class acceptance rates.

  Args:
    initial_probs: The class probabilities of the data.
    target_probs: The desired class proportion in minibatches.
  Returns:
    A list of the per-class acceptance probabilities.

  This method is based on solving the following analysis:

  Let F be the probability of a rejection (on any example).
  Let p_i be the proportion of examples in the data in class i (init_probs)
  Let a_i is the rate the rejection sampler should *accept* class i
  Let t_i is the target proportion in the minibatches for class i (target_probs)

  ```
  F = sum_i(p_i * (1-a_i))
    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1
  ```

  An example with class `i` will be accepted if `k` rejections occur, then an
  example with class `i` is seen by the rejector, and it is accepted. This can
  be written as follows:

  ```
  t_i = sum_k=0^inf(F^k * p_i * a_i)
      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1
      = p_i * a_i / sum_j(p_j * a_j)        using F from above
  ```

  Note that the following constraints hold:
  ```
  0 <= p_i <= 1, sum_i(p_i) = 1
  0 <= a_i <= 1
  0 <= t_i <= 1, sum_i(t_i) = 1
  ```


  A solution for a_i in terms of the other variabes is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```
  """
  # Add tiny to initial_probs to avoid divide by zero.
  denom = (initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny)
  ratio_l = target_probs / denom

  # Calculate list of acceptance probabilities.
  max_ratio = math_ops.reduce_max(ratio_l)
  return ratio_l / max_ratio


def _estimate_data_distribution(c, num_examples_per_class_seen):
  """Estimate data distribution as labels are seen.

  Args:
    c: The class labels.  Type `int32`, shape `[batch_size]`.
    num_examples_per_class_seen: A `ResourceVariable` containing counts.
      Type `int64`, shape `[num_classes]`.

  Returns:
    dist: The updated distribution.  Type `float32`, shape `[num_classes]`.
  """
  num_classes = num_examples_per_class_seen.get_shape()[0].value
  # Update the class-count based on what labels are seen in
  # batch.  But do this asynchronously to avoid performing a
  # cross-device round-trip.  Just use the cached value.
  num_examples_per_class_seen = num_examples_per_class_seen.assign_add(
      math_ops.reduce_sum(
          array_ops.one_hot(c, num_classes, dtype=dtypes.int64), 0))
  init_prob_estimate = math_ops.truediv(
      num_examples_per_class_seen,
      math_ops.reduce_sum(num_examples_per_class_seen))
  return math_ops.cast(init_prob_estimate, dtypes.float32)


def rejection_resample(class_func, target_dist, initial_dist=None, seed=None):
  """A transformation that resamples a dataset to achieve a target distribution.

  **NOTE** Resampling is performed via rejection sampling; some fraction
  of the input values will be dropped.

  Args:
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    target_dist: A floating point type tensor, shaped `[num_classes]`.
    initial_dist: (Optional.)  A floating point type tensor, shaped
      `[num_classes]`.  If not provided, the true class distribution is
      estimated live in a streaming fashion.
    seed: (Optional.) Python integer seed for the resampler.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    dist_estimation_batch_size = 32
    target_dist_t = ops.convert_to_tensor(target_dist, name="initial_dist")
    class_values_ds = dataset.map(class_func)
    if initial_dist is not None:
      initial_dist_t = ops.convert_to_tensor(initial_dist, name="initial_dist")
      acceptance_dist = _calculate_acceptance_probs(initial_dist_t,
                                                    target_dist_t)
      initial_dist_ds = dataset_ops.Dataset.from_tensors(
          initial_dist_t).repeat()
      acceptance_dist_ds = dataset_ops.Dataset.from_tensors(
          acceptance_dist).repeat()
    else:
      num_classes = (target_dist_t.shape[0].value or
                     array_ops.shape(target_dist_t)[0])
      smoothing_constant = 10
      # Disable device functions and colocation constraints so that the variable
      # will be placed with the eventual DT_VARIANT dataset tensor.
      with ops.colocate_with(None, ignore_existing=True):
        num_examples_per_class_seen = resource_variable_ops.ResourceVariable(
            initial_value=array_ops.fill([num_classes],
                                         np.int64(smoothing_constant)),
            trainable=False,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            name="local_class_count",
            dtype=dtypes.int64)

      def update_estimate_and_tile(c):
        return array_ops.tile(
            array_ops.expand_dims(
                _estimate_data_distribution(c, num_examples_per_class_seen), 0),
            [dist_estimation_batch_size, 1])

      initial_dist_ds = (class_values_ds.batch(dist_estimation_batch_size)
                         .map(update_estimate_and_tile).apply(unbatch()))
      acceptance_dist_ds = initial_dist_ds.map(
          lambda initial: _calculate_acceptance_probs(initial, target_dist_t))

    def maybe_warn_on_large_rejection(accept_dist, initial_dist):
      proportion_rejected = math_ops.reduce_sum(
          (1 - accept_dist) * initial_dist)
      return control_flow_ops.cond(
          math_ops.less(proportion_rejected, .5),
          lambda: accept_dist,
          lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
              accept_dist, [proportion_rejected, initial_dist, accept_dist],
              message="Proportion of examples rejected by sampler is high: ",
              summarize=100,
              first_n=10))

    acceptance_dist_ds = (dataset_ops.Dataset.zip((acceptance_dist_ds,
                                                   initial_dist_ds))
                          .map(maybe_warn_on_large_rejection))

    current_probabilities_ds = dataset_ops.Dataset.zip(
        (acceptance_dist_ds, class_values_ds)).map(array_ops.gather)
    filtered_ds = (
        dataset_ops.Dataset.zip((class_values_ds, current_probabilities_ds,
                                 dataset))
        .filter(lambda _1, p, _2: random_ops.random_uniform([], seed=seed) < p))
    return filtered_ds.map(lambda class_value, _, data: (class_value, data))

  return _apply_fn


def batch_and_drop_remainder(batch_size):
  """A batching transformation that omits the final small batch (if present).

  Like @{tf.contrib.data.Dataset.batch}, this transformation combines
  consecutive elements of this dataset into batches. However, if the batch
  size does not evenly divide the input dataset size, this transformation will
  drop the final smaller element.

  The following example illustrates the difference between this
  transformation and `Dataset.batch()`:

  ```python
  dataset = tf.contrib.data.Dataset.range(200)
  batched = dataset.apply(tf.contrib.data.batch_and_drop_remainder(128))
  print(batched.output_shapes)  # ==> "(128,)" (the batch dimension is known)
  ```

  By contrast, `dataset.batch(128)` would yield a two-element dataset with
  shapes `(128,)` and `(72,)`, so the batch dimension would not be statically
  known.

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    tensor_batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")

    batched = dataset.batch(tensor_batch_size)
    flattened = _RestructuredDataset(batched,
                                     tuple(nest.flatten(batched.output_types)))

    def _predicate(*xs):
      """Return `True` if this element is a full batch."""
      # Extract the dynamic batch size from the first component of the flattened
      # batched element.
      first_component = xs[0]
      first_component_batch_size = array_ops.shape(
          first_component, out_type=dtypes.int64)[0]

      return math_ops.equal(first_component_batch_size, tensor_batch_size)

    filtered = flattened.filter(_predicate)

    maybe_constant_batch_size = tensor_util.constant_value(tensor_batch_size)

    def _set_first_dimension(shape):
      return shape.merge_with(
          tensor_shape.vector(maybe_constant_batch_size).concatenate(shape[1:]))

    known_shapes = nest.map_structure(_set_first_dimension,
                                      batched.output_shapes)
    return _RestructuredDataset(filtered, batched.output_types, known_shapes)

  return _apply_fn


def read_batch_features(file_pattern,
                        batch_size,
                        features,
                        reader,
                        reader_args=None,
                        randomize_input=True,
                        num_epochs=None,
                        capacity=10000):
  """Reads batches of Examples.

  Example:

  ```
  serialized_examples = [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "code", "art" ] } } }
    },
    features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
      feature { key: "kws" value { bytes_list { value: [ "sports" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
    "kws": VarLenFeature(dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
    "kws": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["code", "art", "sports"]
      dense_shape=[2, 2]),
  }
  ```

  Args:
    file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.gfile.Glob` for pattern rules.
    batch_size: An int representing the number of consecutive elements of this
      dataset to combine in a single batch.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. See `tf.parse_example`.
    reader: A function or class that can be called with a `filenames` tensor
      and (optional) `reader_args` and returns a `Dataset` of serialized
      Examples.
    reader_args: Additional arguments to pass to the reader class.
    randomize_input: Whether the input should be randomized.
    num_epochs: Integer specifying the number of times to read through the
      dataset. If None, cycles through the dataset forever.
    capacity: Capacity of the ShuffleDataset. A large capacity ensures better
      shuffling but would increase memory usage and startup time.

  Returns:
    A dict from keys in features to Tensor or SparseTensor objects.
  """
  filenames = _get_file_names(file_pattern, randomize_input)
  if reader_args:
    dataset = reader(filenames, *reader_args)
  else:
    dataset = reader(filenames)
  if dataset.output_types == (dtypes.string, dtypes.string):
    dataset = dataset.map(lambda unused_k, v: v)
  elif dataset.output_types != dtypes.string:
    raise TypeError("`reader` must be a dataset of `tf.string` values, "
                    "or `(tf.string, tf.string)` key-value pairs.")
  if num_epochs != 1:
    dataset = dataset.repeat(num_epochs)
  if randomize_input:
    dataset = dataset.shuffle(capacity)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda x: _parse_example(x, features))
  iterator = dataset.make_one_shot_iterator()
  outputs = iterator.get_next()
  index = 0
  result = {}
  for key in sorted(features.keys()):
    feature = features[key]
    if isinstance(feature, parsing_ops.FixedLenFeature):
      result[key] = outputs[index]
      index += 1
    else:
      result[key] = sparse_tensor_lib.SparseTensor(
          indices=outputs[index],
          values=outputs[index + 1],
          dense_shape=outputs[index + 2])
      index += 3
  return result


def _parse_example(serialized, features):
  parsed = parsing_ops.parse_example(serialized, features)
  result = []
  for key in sorted(features.keys()):
    val = parsed[key]
    if isinstance(val, sparse_tensor_lib.SparseTensor):
      result.extend([val.indices, val.values, val.dense_shape])
    else:
      result.append(val)
  return tuple(result)


def _get_file_names(file_pattern, randomize_input):
  """Parse list of file names from pattern, optionally shuffled.

  Args:
    file_pattern: File glob pattern, or list of glob patterns.
    randomize_input: Whether to shuffle the order of file names.

  Returns:
    List of file names matching `file_pattern`.

  Raises:
    ValueError: If `file_pattern` is empty, or pattern matches no files.
  """
  if isinstance(file_pattern, list):
    if not file_pattern:
      raise ValueError("File pattern is empty.")
    file_names = []
    for entry in file_pattern:
      file_names.extend(gfile.Glob(entry))
  else:
    file_names = list(gfile.Glob(file_pattern))

  if not file_names:
    raise ValueError("No files match %s." % file_pattern)

  # Sort files so it will be deterministic for unit tests.
  if not randomize_input:
    file_names = sorted(file_names)
  return file_names


class DenseToSparseBatchDataset(dataset_ops.Dataset):
  """A `Dataset` that batches ragged dense elements into `tf.SparseTensor`s."""

  def __init__(self, input_dataset, batch_size, row_shape):
    """See `Dataset.dense_to_sparse_batch()` for more details."""
    super(DenseToSparseBatchDataset, self).__init__()
    if not isinstance(input_dataset.output_types, dtypes.DType):
      raise TypeError("DenseToSparseDataset requires an input whose elements "
                      "have a single component, whereas the input has %r." %
                      input_dataset.output_types)
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    # pylint: disable=protected-access
    self._row_shape = dataset_ops._partial_shape_to_tensor(row_shape)
    # pylint: enable=protected-access

  def make_dataset_resource(self):
    return gen_dataset_ops.dense_to_sparse_batch_dataset(
        self._input_dataset.make_dataset_resource(),
        self._batch_size,
        self._row_shape,
        output_shapes=self.output_shapes,
        output_types=self.output_types)

  @property
  def output_shapes(self):
    num_elements = tensor_shape.Dimension(None)
    return (tensor_shape.matrix(num_elements, self._row_shape.shape[0] + 1),
            tensor_shape.vector(num_elements),
            tensor_shape.vector(self._row_shape.shape[0] + 1))

  @property
  def output_types(self):
    return (dtypes.int64, self._input_dataset.output_types, dtypes.int64)


class _RestructuredDataset(dataset_ops.Dataset):
  """An internal helper for changing the structure and shape of a dataset."""

  def __init__(self, dataset, output_types, output_shapes=None):
    """Creates a new dataset with the given output types and shapes.

    The given `dataset` must have a structure that is convertible:
    * `dataset.output_types` must be the same as `output_types` module nesting.
    * Each shape in `dataset.output_shapes` must be compatible with each shape
      in `output_shapes` (if given).

    Note: This helper permits "unsafe casts" for shapes, equivalent to using
    `tf.Tensor.set_shape()` where domain-specific knowledge is available.

    Args:
      dataset: A `Dataset` object.
      output_types: A nested structure of `tf.DType` objects.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects.
        If omitted, the shapes will be inherited from `dataset`.

    Raises:
      ValueError: If either `output_types` or `output_shapes` is not compatible
        with the structure of `dataset`.
    """
    super(_RestructuredDataset, self).__init__()
    self._dataset = dataset

    # Validate that the types are compatible.
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    flat_original_types = nest.flatten(dataset.output_types)
    flat_new_types = nest.flatten(output_types)
    if flat_original_types != flat_new_types:
      raise ValueError(
          "Dataset with output types %r cannot be restructured to have output "
          "types %r" % (dataset.output_types, output_types))

    self._output_types = output_types

    if output_shapes is None:
      # Inherit shapes from the original `dataset`.
      self._output_shapes = nest.pack_sequence_as(output_types,
                                                  nest.flatten(
                                                      dataset.output_shapes))
    else:
      # Validate that the shapes are compatible.
      nest.assert_same_structure(output_types, output_shapes)
      flat_original_shapes = nest.flatten(dataset.output_shapes)
      flat_new_shapes = nest.flatten_up_to(output_types, output_shapes)

      for original_shape, new_shape in zip(flat_original_shapes,
                                           flat_new_shapes):
        if not original_shape.is_compatible_with(new_shape):
          raise ValueError(
              "Dataset with output shapes %r cannot be restructured to have "
              "incompatible output shapes %r" % (dataset.output_shapes,
                                                 output_shapes))
      self._output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)

  def make_dataset_resource(self):
    return self._dataset.make_dataset_resource()

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes
