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
"""Python wrappers for Datasets and Iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from tensorflow.contrib.data.python.framework import function
from tensorflow.contrib.data.python.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import gfile


class Iterator(object):
  """Represents the state of iterating through a `Dataset`."""

  def __init__(self, iterator_resource, initializer, output_types,
               output_shapes):
    """Creates a new iterator from the given iterator resource.

    NOTE(mrry): Most users will not call this initializer directly, and will
    instead use `Iterator.from_dataset()` or `Dataset.make_one_shot_iterator()`.

    Args:
      iterator_resource: A `tf.resource` scalar `tf.Tensor` representing the
        iterator.
      initializer: A `tf.Operation` that should be run to initialize this
        iterator.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this iterator.
      output_shapes: A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset.
    """
    self._iterator_resource = iterator_resource
    self._initializer = initializer
    self._output_types = output_types
    self._output_shapes = output_shapes

  @staticmethod
  def from_dataset(dataset, shared_name=None):
    """Creates a new, uninitialized `Iterator` from the given `Dataset`.

    To initialize this iterator, you must run its `initializer`:

    ```python
    dataset = ...
    iterator = Iterator.from_dataset(dataset)
    # ...
    sess.run(iterator.initializer)
    ```

    Args:
      dataset: A `Dataset` object.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).

    Returns:
      An `Iterator`.
    """
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator(
        container="",
        shared_name=shared_name,
        output_types=nest.flatten(dataset.output_types),
        output_shapes=nest.flatten(dataset.output_shapes))
    initializer = gen_dataset_ops.make_iterator(dataset.make_dataset_resource(),
                                                iterator_resource)
    return Iterator(iterator_resource, initializer, dataset.output_types,
                    dataset.output_shapes)

  @staticmethod
  def from_structure(output_types, output_shapes=None, shared_name=None):
    """Creates a new, uninitialized `Iterator` with the given structure.

    This iterator-constructing method can be used to create an iterator that
    is reusable with many different datasets.

    The returned iterator is not bound to a particular dataset, and it has
    no `initializer`. To initialize the iterator, run the operation returned by
    `Iterator.make_initializer(dataset)`.

    The following is an example

    ```python
    iterator = Iterator.from_structure(tf.int64, tf.TensorShape([]))

    dataset_range = Dataset.range(10)
    range_initializer = iterator.make_initializer(dataset_range)

    dataset_evens = dataset_range.filter(lambda x: x % 2 == 0)
    evens_initializer = iterator.make_initializer(dataset_evens)

    # Define a model based on the iterator; in this example, the model_fn
    # is expected to take scalar tf.int64 Tensors as input (see
    # the definition of 'iterator' above).
    prediction, loss = model_fn(iterator.get_next())

    # Train for `num_epochs`, where for each epoch, we first iterate over
    # dataset_range, and then iterate over dataset_evens.
    for _ in range(num_epochs):
      # Initialize the iterator to `dataset_range`
      sess.run(range_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break

      # Initialize the iterator to `dataset_evens`
      sess.run(evens_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break
    ```

    Args:
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this iterator.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset. If
        omitted, each component will have an unconstrainted shape.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).

    Returns:
      An `Iterator`.

    Raises:
      TypeError: If the structures of `output_shapes` and `output_types` are
        not the same.
    """
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)
    nest.assert_same_structure(output_types, output_shapes)
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator(
        container="",
        shared_name=shared_name,
        output_types=nest.flatten(output_types),
        output_shapes=nest.flatten(output_shapes))
    return Iterator(iterator_resource, None, output_types, output_shapes)

  @property
  def initializer(self):
    """A `tf.Operation` that should be run to initialize this iterator.

    Returns:
      A `tf.Operation` that should be run to initialize this iterator

    Raises:
      ValueError: If this iterator initializes itself automatically.
    """
    if self._initializer is not None:
      return self._initializer
    else:
      # TODO(mrry): Consider whether one-shot iterators should have
      # initializers that simply reset their state to the beginning.
      raise ValueError("Iterator does not have an initializer.")

  def make_initializer(self, dataset):
    """Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` with compatible structure to this iterator.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        element structure.
    """
    nest.assert_same_structure(self._output_types, dataset.output_types)
    nest.assert_same_structure(self._output_shapes, dataset.output_shapes)
    for iterator_dtype, dataset_dtype in zip(
        nest.flatten(self._output_types), nest.flatten(dataset.output_types)):
      if iterator_dtype != dataset_dtype:
        raise TypeError(
            "Expected output types %r but got dataset with output types %r." %
            (self._output_types, dataset.output_types))
    for iterator_shape, dataset_shape in zip(
        nest.flatten(self._output_shapes), nest.flatten(dataset.output_shapes)):
      if not iterator_shape.is_compatible_with(dataset_shape):
        raise TypeError("Expected output shapes compatible with %r but got "
                        "dataset with output shapes %r." %
                        (self._output_shapes, dataset.output_shapes))
    return gen_dataset_ops.make_iterator(dataset.make_dataset_resource(),
                                         self._iterator_resource)

  def get_next(self, name=None):
    """Returns a nested structure of `tf.Tensor`s containing the next element.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    return nest.pack_sequence_as(
        self._output_types,
        gen_dataset_ops.iterator_get_next(
            self._iterator_resource,
            output_types=nest.flatten(self._output_types),
            output_shapes=nest.flatten(self._output_shapes),
            name=name))

  def dispose_op(self, name=None):
    """Returns a `tf.Operation` that destroys this iterator.

    The returned operation may be used to release any resources consumed by
    this iterator without closing the session.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation`.
    """
    return gen_dataset_ops.iterator_dispose(self._iterator_resource, name=name)

  @property
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this iterator.
    """
    return self._output_shapes

  @property
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this iterator.
    """
    return self._output_types


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


class Dataset(object):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements (nested structures of tensors) and a "logical
  plan" of transformations that act on those elements.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  @abc.abstractmethod
  def make_dataset_resource(self):
    """Creates a `tf.Tensor` of  `tf.resource` tensor representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.resource` type, which represents this dataset.
    """
    raise NotImplementedError("Dataset.make_dataset_resource")

  def make_initializable_iterator(self, shared_name=None):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    **N.B.** The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it.

    Args:
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).


    Returns:
      An `Iterator` over the elements of this dataset.
    """
    return Iterator.from_dataset(self, shared_name)

  def make_one_shot_iterator(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    **N.B.** The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.
    """
    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(capture_by_value=True)
    def _make_dataset():
      return self.make_dataset_resource()

    _make_dataset.add_to_graph(ops.get_default_graph())

    return Iterator(
        gen_dataset_ops.one_shot_iterator(
            dataset_factory=_make_dataset,
            output_types=nest.flatten(self.output_types),
            output_shapes=nest.flatten(self.output_shapes)), None,
        self.output_types, self.output_shapes)

  @abc.abstractproperty
  def output_shapes(self):
    """Returns the shape of each component of an element of this dataset.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    raise NotImplementedError("Dataset.output_shapes")

  @abc.abstractproperty
  def output_types(self):
    """Returns the type of each component of an element of this dataset.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    raise NotImplementedError("Dataset.output_types")

  def __repr__(self):
    output_shapes = nest.map_structure(str, self.output_shapes)
    output_shapes = str(output_shapes).replace("'", "")
    output_types = nest.map_structure(repr, self.output_types)
    output_types = str(output_types).replace("'", "")
    return ("<%s shapes: %s, types: %s>" % (type(self).__name__, output_shapes,
                                            output_types))

  @staticmethod
  def from_tensors(tensors):
    """Creates a `Dataset` with a single element, comprising the given tensors.

    Args:
      tensors: A nested structure of tensors.

    Returns:
      A `Dataset`.
    """
    return TensorDataset(tensors)

  @staticmethod
  def from_tensor_slices(tensors):
    """Creates a `Dataset` whose elements are slices of the given tensors.

    Args:
      tensors: A nested structure of tensors, each having the same size in the
        0th dimension.

    Returns:
      A `Dataset`.
    """
    return TensorSliceDataset(tensors)

  @staticmethod
  def from_sparse_tensor_slices(sparse_tensor):
    """Splits each rank-N `tf.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.SparseTensor`.

    Returns:
      A `Dataset` of rank-(N-1) sparse tensors.
    """
    return SparseTensorSliceDataset(sparse_tensor)

  @staticmethod
  def range(*args):
    """Creates a `Dataset` of a step-separated range of values.

    For example:

    ```python
    Dataset.range(5) == [0, 1, 2, 3, 4]
    Dataset.range(2, 5) == [2, 3, 4]
    Dataset.range(1, 5, 2) == [1, 3]
    Dataset.range(1, 5, -2) == []
    Dataset.range(5, 1) == []
    Dataset.range(5, 1, -2) == [5, 3]
    ```

    Args:
      *args: follow same semantics as python's xrange.
        len(args) == 1 -> start = 0, stop = args[0], step = 1
        len(args) == 2 -> start = args[0], stop = args[1], step = 1
        len(args) == 3 -> start = args[0], stop = args[1, stop = args[2]

    Returns:
      A `RangeDataset`.

    Raises:
      ValueError: if len(args) == 0.
    """
    return RangeDataset(*args)

  @staticmethod
  def zip(datasets):
    """Creates a `Dataset` by zipping together the given datasets.

    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be an arbitrary nested structure of `Dataset` objects.
    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3 }
    b = { 4, 5, 6 }
    c = { (7, 8), (9, 10), (11, 12) }
    d = { 13, 14 }

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
    Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }

    # The `datasets` argument may contain an arbitrary number of
    # datasets.
    Dataset.zip((a, b, c)) == { (1, 4, (7, 8)),
                                (2, 5, (9, 10)),
                                (3, 6, (11, 12)) }

    # The number of elements in the resulting dataset is the same as
    # the size of the smallest dataset in `datasets`.
    Dataset.zip((a, d)) == { (1, 13), (2, 14) }
    ```

    Args:
      datasets: A nested structure of datasets.

    Returns:
      A `Dataset`.
    """
    return ZipDataset(datasets)

  @staticmethod
  def read_batch_features(file_pattern,
                          batch_size,
                          features,
                          reader,
                          reader_args=None,
                          randomize_input=True,
                          num_epochs=None,
                          capacity=10000):
    """Reads batches of Examples.

    Args:
      file_pattern: A string pattern or a placeholder with list of filenames.
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      features: A `dict` mapping feature keys to `FixedLenFeature` or
        `VarLenFeature` values. See `tf.parse_example`.
      reader: A function or class that can be called with a `filenames` tensor
        and (optional) `reader_args` and returns a `Dataset` of serialized
        Examples.
      reader_args: Additional arguments to pass to the reader class.
      randomize_input: Whether the input should be randomized.
      num_epochs: Integer specifying the number of times to read through the
        dataset. If None, cycles through the dataset forever.
      capacity: Capacity of the ShuffleDataset.

    Returns:
      A `Dataset`.
    """
    if isinstance(file_pattern, str):
      filenames = _get_file_names(file_pattern, randomize_input)
    else:
      filenames = file_pattern
    if reader_args:
      dataset = reader(filenames, *reader_args)
    else:
      dataset = reader(filenames)
    dataset = dataset.repeat(num_epochs)
    if randomize_input:
      dataset = dataset.shuffle(capacity)
    dataset = dataset.map(lambda x: _parse_example(nest.flatten(x), features))
    dataset = dataset.batch(batch_size)
    return dataset

  @staticmethod
  def list_files(file_pattern):
    """A dataset of all files matching a pattern.

    Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset would
      produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py

    Args:
      file_pattern: A string or scalar string `tf.Tensor`, representing
        the filename pattern that will be matched.

    Returns:
     A `Dataset` of strings corresponding to file names.
    """
    return Dataset.from_tensor_slices(gen_io_ops.matching_files(file_pattern))

  def repeat(self, count=None):
    """Repeats this dataset `count` times.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the elements of this dataset should be repeated. The
        default behavior (if `count` is `None` or `-1`) is for the elements to
        be repeated indefinitely.

    Returns:
      A `Dataset`.
    """
    return RepeatDataset(self, count)

  def enumerate(self, start=0):
    """Enumerate the elements of this dataset.  Similar to python's `enumerate`.

    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3 }
    b = { (7, 8), (9, 10), (11, 12) }

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    a.enumerate(start=5) == { (5, 1), (6, 2), (7, 3) }
    b.enumerate() == { (0, (7, 8)), (1, (9, 10)), (2, (11, 12)) }
    ```

    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start
        value for enumeration.

    Returns:
      A `Dataset`.
    """
    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    return Dataset.zip((Dataset.range(start, max_value), self))

  def shuffle(self, buffer_size, seed=None):
    """Randomly shuffles the elements of this dataset.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        random seed that will be used to create the distribution. See
        @{tf.set_random_seed} for behavior.

    Returns:
      A `Dataset`.
    """
    return ShuffleDataset(self, buffer_size, seed)

  def cache(self, filename=""):
    """Caches the elements in this dataset.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching tensors in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.

    Returns:
      A `Dataset`.
    """
    return CacheDataset(self, filename)

  def take(self, count):
    """Creates a `Dataset` with at most `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be taken to form the new dataset.
        If `count` is -1, or if `count` is greater than the size of this
        dataset, the new dataset will contain all elements of this dataset.

    Returns:
      A `Dataset`.
    """
    return TakeDataset(self, count)

  def skip(self, count):
    """Creates a `Dataset` that skips `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number
        of elements of this dataset that should be skipped to form the
        new dataset.  If `count` is greater than the size of this
        dataset, the new dataset will contain no elements.  If `count`
        is -1, skips the entire dataset.

    Returns:
      A `Dataset`.
    """
    return SkipDataset(self, count)

  def ignore_errors(self):
    """Creates a `Dataset` from this one and silently ignores any errors.

    Use this transformation to produce a dataset that contains the same elements
    as the input, but silently drops any elements that caused an error. For
    example:

    ```python
    dataset = tf.contrib.data.Dataset.from_tensor_slices([1., 2., 0., 4.])

    # Computing `tf.check_numerics(1. / 0.)` will raise an InvalidArgumentError.
    dataset = dataset.map(lambda x: tf.check_numerics(1. / x, "error"))

    # Using `ignore_errors()` will drop the element that causes an error.
    dataset = dataset.ignore_errors()  # ==> { 1., 0.5, 0.2 }
    ```

    Returns:
      A `Dataset`.
    """
    return IgnoreErrorsDataset(self)

  def batch(self, batch_size):
    """Combines consecutive elements of this dataset into batches.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.

    Returns:
      A `Dataset`.
    """
    return BatchDataset(self, batch_size)

  def padded_batch(self, batch_size, padded_shapes, padding_values=None):
    """Combines consecutive elements of this dataset into padded batches.

    Like `Dataset.dense_to_sparse_batch()`, this method combines
    multiple consecutive elements of this dataset, which might have
    different shapes, into a single element. The tensors in the
    resulting element have an additional outer dimension, and are
    padded to the respective shape in `padded_shapes`.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: A nested structure of `tf.TensorShape` or
        `tf.int64` vector tensor-like objects representing the shape
        to which the respective component of each input element should
        be padded prior to batching. Any unknown dimensions
        (e.g. `tf.Dimension(None)` in a `tf.TensorShape` or `-1` in a
        tensor-like object) will be padded to the maximum size of that
        dimension in each batch.
      padding_values: (Optional.) A nested structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the
        respective components.  Defaults are `0` for numeric types and
        the empty string for string types.

    Returns:
      A `Dataset`.
    """
    return PaddedBatchDataset(self, batch_size, padded_shapes, padding_values)

  def dense_to_sparse_batch(self, batch_size, row_shape):
    """Batches ragged elements of this dataset into `tf.SparseTensor`s.

    Like `Dataset.padded_batch()`, this method combines multiple
    consecutive elements of this dataset, which might have different
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

    a.dense_to_sparse_batch(batch_size=2, row_shape=[6]) == {
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
      A `Dataset`.
    """
    return DenseToSparseBatchDataset(self, batch_size, row_shape)

  def group_by_window(self, key_func, reduce_func, window_size):
    """Performs a windowed "group-by" operation on this dataset.

    This method maps each consecutive element in this dataset to a key
    using `key_func` and groups the elements by key. It then applies
    `reduce_func` to at most `window_size` elements matching the same
    key. All execpt the final window for each key will contain
    `window_size` elements; the final window may be smaller.

    Args:
      key_func: A function mapping a nested structure of tensors
        (having shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.int64` tensor.
      reduce_func: A function mapping a key and a dataset of up to `batch_size`
        consecutive elements matching that key to another dataset.
      window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements matching the same key to combine in a single
        batch, which will be passed to `reduce_func`.

    Returns:
      A `Dataset`.
    """
    return GroupByWindowDataset(self, key_func, reduce_func, window_size)

  def map(self, map_func, num_threads=None, output_buffer_size=None):
    """Maps `map_func` across this datset.

    Args:
      map_func: A function mapping a nested structure of tensors (having
        shapes and types defined by `self.output_shapes` and
       `self.output_types`) to another nested structure of tensors.
      num_threads: (Optional.) A `tf.int32` scalar `tf.Tensor`, representing
        the number of threads to use for processing elements in parallel. If
        not specified, elements will be processed sequentially without
        buffering.
      output_buffer_size: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the maximum number of processed elements that will be
        buffered when processing in parallel.

    Returns:
      A `Dataset`.
    """
    return MapDataset(self, map_func, num_threads, output_buffer_size)

  def flat_map(self, map_func):
    """Maps `map_func` across this dataset and flattens the result.

    Args:
      map_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        `Dataset`.

    Returns:
      A `Dataset`.
    """
    return FlatMapDataset(self, map_func)

  def unbatch(self):
    """Splits elements of this dataset into sequences of consecutive elements.

    For example, if elements of this dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary from element to element, then for each element in
    this dataset, the unbatched dataset will contain `B` consecutive elements
    of shape `[a0, a1, ...]`.

    Returns:
      A `Dataset`.
    """
    return self.flat_map(
        map_func=lambda *args: Dataset.from_tensor_slices(args))

  def filter(self, predicate):
    """Filters this dataset according to `predicate`.

    Args:
      predicate: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        scalar `tf.bool` tensor.

    Returns:
      A `Dataset`.
    """
    return FilterDataset(self, predicate)


class TensorDataset(Dataset):
  """A `Dataset` with a single element, viz. a nested structure of tensors."""

  def __init__(self, tensors):
    """See `Dataset.from_tensors()` for details."""
    super(TensorDataset, self).__init__()
    with ops.name_scope("tensors"):
      self._tensors = nest.pack_sequence_as(tensors, [
          ops.convert_to_tensor(t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(tensors))
      ])

  def make_dataset_resource(self):
    return gen_dataset_ops.tensor_dataset(
        nest.flatten(self._tensors),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return nest.pack_sequence_as(self._tensors,
                                 [t.shape for t in nest.flatten(self._tensors)])

  @property
  def output_types(self):
    return nest.pack_sequence_as(self._tensors,
                                 [t.dtype for t in nest.flatten(self._tensors)])


class TensorSliceDataset(Dataset):
  """A `Dataset` of slices from a nested structure of tensors."""

  def __init__(self, tensors):
    """See `Dataset.from_tensor_slices()` for details."""
    super(TensorSliceDataset, self).__init__()
    with ops.name_scope("tensors"):
      flat_tensors = [
          ops.convert_to_tensor(t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(tensors))
      ]

    self._tensors = nest.pack_sequence_as(tensors, flat_tensors)
    batch_dim = flat_tensors[0].get_shape()[0]
    for t in flat_tensors[1:]:
      batch_dim.assert_is_compatible_with(t.get_shape()[0])

  def make_dataset_resource(self):
    return gen_dataset_ops.tensor_slice_dataset(
        nest.flatten(self._tensors),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return nest.pack_sequence_as(self._tensors, [
        tensor_shape.TensorShape(t.shape[1:])
        for t in nest.flatten(self._tensors)
    ])

  @property
  def output_types(self):
    return nest.pack_sequence_as(self._tensors,
                                 [t.dtype for t in nest.flatten(self._tensors)])


class SparseTensorSliceDataset(Dataset):
  """A `Dataset` that splits a rank-N `tf.SparseTensor` into its rows."""

  def __init__(self, sparse_tensor):
    """See `Dataset.from_sparse_tensor_slices()` for details."""
    super(SparseTensorSliceDataset, self).__init__()
    if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
      raise TypeError("`sparse_tensor` must be a `tf.SparseTensor` object.")
    self._sparse_tensor = sparse_tensor

  def make_dataset_resource(self):
    return gen_dataset_ops.sparse_tensor_slice_dataset(
        self._sparse_tensor.indices, self._sparse_tensor.values,
        self._sparse_tensor.dense_shape)

  @property
  def output_shapes(self):
    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape[1] - 1).merge_with(shape_shape[0] - 1)
    num_values = tensor_shape.Dimension(None)
    return (tensor_shape.TensorShape([num_values, rank]),
            tensor_shape.TensorShape([num_values]), tensor_shape.TensorShape(
                [rank]))

  @property
  def output_types(self):
    return (dtypes.int64, self._sparse_tensor.dtype, dtypes.int64)


class ZipDataset(Dataset):
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets):
    """See `Dataset.zip()` for details."""
    super(ZipDataset, self).__init__()
    self._datasets = datasets

  def make_dataset_resource(self):
    return gen_dataset_ops.zip_dataset(
        [ds.make_dataset_resource() for ds in nest.flatten(self._datasets)],
        output_shapes=[
            s
            for ds in nest.flatten(self._datasets)
            for s in nest.flatten(ds.output_shapes)
        ],
        output_types=[
            t
            for ds in nest.flatten(self._datasets)
            for t in nest.flatten(ds.output_types)
        ])

  @property
  def output_shapes(self):
    return nest.pack_sequence_as(self._datasets, [
        ds.output_shapes for ds in nest.flatten(self._datasets)
    ])

  @property
  def output_types(self):
    return nest.pack_sequence_as(self._datasets, [
        ds.output_types for ds in nest.flatten(self._datasets)
    ])


class RepeatDataset(Dataset):
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count):
    """See `Dataset.repeat()` for details."""
    super(RepeatDataset, self).__init__()
    self._input_dataset = input_dataset
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")

  def make_dataset_resource(self):
    return gen_dataset_ops.repeat_dataset(
        self._input_dataset.make_dataset_resource(),
        count=self._count,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class RangeDataset(Dataset):
  """A `Dataset` of a step separated range of values."""

  def __init__(self, *args):
    """See `Dataset.range()` for details."""
    super(RangeDataset, self).__init__()
    self._parse_args(*args)

  def _parse_args(self, *args):
    if len(args) == 1:
      self._start = self._build_tensor(0, "start")
      self._stop = args[0]
      self._step = self._build_tensor(1, "step")
    elif len(args) == 2:
      self._start = args[0]
      self._stop = args[1]
      self._step = self._build_tensor(1, "step")
    elif len(args) == 3:
      self._start = args[0]
      self._stop = args[1]
      self._step = args[2]
    else:
      raise ValueError("Invalid arguments to RangeDataset: %s" % str(args))

  def _build_tensor(self, int64_value, name):
    return constant_op.constant(int64_value, dtype=dtypes.int64, name=name)

  def make_dataset_resource(self):
    return gen_dataset_ops.range_dataset(
        start=self._start,
        stop=self._stop,
        step=self._step,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.int64


class CacheDataset(Dataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename):
    """See `Dataset.cache()` for details."""
    super(CacheDataset, self).__init__()
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")

  def make_dataset_resource(self):
    return gen_dataset_ops.cache_dataset(
        self._input_dataset.make_dataset_resource(),
        filename=self._filename,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class ShuffleDataset(Dataset):
  """A `Dataset` that randomly shuffles the elements of its input."""

  def __init__(self, input_dataset, buffer_size, seed=None):
    """See `Dataset.shuffle()` for details."""
    super(ShuffleDataset, self).__init__()
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    seed, seed2 = random_seed.get_seed(seed)
    if seed is None:
      self._seed = constant_op.constant(0, dtype=dtypes.int64, name="seed")
    else:
      self._seed = ops.convert_to_tensor(seed, dtype=dtypes.int64, name="seed")
    if seed2 is None:
      self._seed2 = constant_op.constant(0, dtype=dtypes.int64, name="seed2")
    else:
      self._seed2 = ops.convert_to_tensor(
          seed2, dtype=dtypes.int64, name="seed2")

  def make_dataset_resource(self):
    return gen_dataset_ops.shuffle_dataset(
        self._input_dataset.make_dataset_resource(),
        buffer_size=self._buffer_size,
        seed=self._seed,
        seed2=self._seed2,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class TakeDataset(Dataset):
  """A `Dataset` containing the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.take()` for details."""
    super(TakeDataset, self).__init__()
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")

  def make_dataset_resource(self):
    return gen_dataset_ops.take_dataset(
        self._input_dataset.make_dataset_resource(),
        count=self._count,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class SkipDataset(Dataset):
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.skip()` for details."""
    super(SkipDataset, self).__init__()
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")

  def make_dataset_resource(self):
    return gen_dataset_ops.skip_dataset(
        self._input_dataset.make_dataset_resource(),
        count=self._count,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class IgnoreErrorsDataset(Dataset):
  """A `Dataset` that silently ignores errors when computing its input."""

  def __init__(self, input_dataset):
    """See `Dataset.ignore_errors()` for details."""
    super(IgnoreErrorsDataset, self).__init__()
    self._input_dataset = input_dataset

  def make_dataset_resource(self):
    return gen_dataset_ops.ignore_errors_dataset(
        self._input_dataset.make_dataset_resource(),
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class BatchDataset(Dataset):
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size):
    """See `Dataset.batch()` for details."""
    super(BatchDataset, self).__init__()
    self._input_dataset = input_dataset
    self._batch_size = batch_size

  def make_dataset_resource(self):
    return gen_dataset_ops.batch_dataset(
        self._input_dataset.make_dataset_resource(),
        batch_size=self._batch_size,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

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


def _partial_shape_to_tensor(shape_like):
  try:
    # First attempt to convert the input to a shape, and return the
    # "canonical" tensor representation, which uses `-1` in place of
    # `None`.
    shape_like = tensor_shape.as_shape(shape_like)
    return ops.convert_to_tensor(
        [dim if dim is not None else -1 for dim in shape_like.as_list()],
        dtype=dtypes.int64)
  except (TypeError, ValueError):
    # The argument was not trivially convertible to a
    # `tf.TensorShape`, so fall back on the conversion to tensor
    # machinery.
    return ops.convert_to_tensor(shape_like, dtype=dtypes.int64)


def _padding_value_to_tensor(value, output_type):
  """Converts the padding value to a tensor.

  Args:
    value: The padding value.
    output_type: Its expected dtype.

  Returns:
    A scalar `Tensor`.

  Raises:
    ValueError: if the padding value is not a scalar.
    TypeError: if the padding value's type does not match `output_type`.
  """
  value = ops.convert_to_tensor(value, name="padding_value")
  if not value.shape.is_compatible_with(tensor_shape.scalar()):
    raise ValueError("Padding value should be a scalar, but is not: %s" % value)
  if value.dtype != output_type:
    raise TypeError("Padding value tensor (%s) does not match output type: %s" %
                    (value, output_type))
  return value


class PaddedBatchDataset(Dataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values):
    """See `Dataset.batch()` for details."""
    super(PaddedBatchDataset, self).__init__()
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    padding_values = (padding_values if padding_values is not None else
                      self._default_padding(input_dataset))
    self._padded_shapes = nest.map_structure_up_to(
        input_dataset.output_shapes, _partial_shape_to_tensor, padded_shapes)
    self._padding_values = nest.map_structure_up_to(
        input_dataset.output_shapes, _padding_value_to_tensor, padding_values,
        input_dataset.output_types)

  def _default_padding(self, input_dataset):

    def make_zero(t):
      if t.base_dtype == dtypes.string:
        return ""
      else:
        return np.zeros_like(t.as_numpy_dtype())

    return nest.map_structure(make_zero, input_dataset.output_types)

  def make_dataset_resource(self):
    return gen_dataset_ops.padded_batch_dataset(
        self._input_dataset.make_dataset_resource(),
        batch_size=self._batch_size,
        padded_shapes=[
            ops.convert_to_tensor(s, dtype=dtypes.int64)
            for s in nest.flatten(self._padded_shapes)
        ],
        padding_values=nest.flatten(self._padding_values),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):

    def _padded_shape_to_batch_shape(s):
      return tensor_shape.vector(None).concatenate(
          tensor_util.constant_value_as_shape(s))

    return nest.map_structure(_padded_shape_to_batch_shape, self._padded_shapes)

  @property
  def output_types(self):
    return self._input_dataset.output_types


class DenseToSparseBatchDataset(Dataset):
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
    self._row_shape = _partial_shape_to_tensor(row_shape)

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


def _should_unpack_args(args):
  """Returns `True` if `args` should be `*args` when passed to a callable."""
  return nest.is_sequence(args) and not isinstance(args, dict)


class _ResourceDataset(Dataset):
  """A Dataset wrapper for a tf.resource-typed function argument."""

  def __init__(self, dataset_resource, output_types, output_shapes):
    super(_ResourceDataset, self).__init__()
    self._dataset_resource = dataset_resource,
    self._output_types = output_types
    self._output_shapes = output_shapes

  def make_dataset_resource(self):
    return self._dataset_resource

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class GroupByWindowDataset(Dataset):
  """A `Dataset` that groups its input and performs a windowed reduction."""

  def __init__(self, input_dataset, key_func, reduce_func, window_size):
    """See `Dataset.group_by_window()` for details."""
    super(GroupByWindowDataset, self).__init__()
    self._input_dataset = input_dataset
    self._window_size = window_size

    @function.Defun(*nest.flatten(input_dataset.output_types))
    def tf_key_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      for arg, shape in zip(args, nest.flatten(input_dataset.output_shapes)):
        arg.set_shape(shape)
      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)
      if _should_unpack_args(nested_args):
        ret = key_func(*nested_args)
      else:
        ret = key_func(nested_args)
      ret = ops.convert_to_tensor(ret, dtype=dtypes.int64)
      if ret.dtype != dtypes.int64:
        raise ValueError("`key_func` must return a single tf.int64 tensor.")
      return ret

    self._key_func = tf_key_func
    self._key_func.add_to_graph(ops.get_default_graph())

    @function.Defun(dtypes.int64, dtypes.resource)
    def tf_reduce_func(key, window_dataset_resource):
      """A wrapper for Defun that facilitates shape inference."""
      key.set_shape([])
      window_dataset = _ResourceDataset(window_dataset_resource,
                                        input_dataset.output_types,
                                        input_dataset.output_shapes)
      output_dataset = reduce_func(key, window_dataset)
      if not isinstance(output_dataset, Dataset):
        raise TypeError("`reduce_func` must return a `Dataset` object.")
      self._output_types = output_dataset.output_types
      self._output_shapes = output_dataset.output_shapes
      return output_dataset.make_dataset_resource()

    self._reduce_func = tf_reduce_func
    self._reduce_func.add_to_graph(ops.get_default_graph())

  def make_dataset_resource(self):
    return gen_dataset_ops.group_by_window_dataset(
        self._input_dataset.make_dataset_resource(),
        self._key_func.captured_inputs,
        self._reduce_func.captured_inputs,
        self._window_size,
        key_func=self._key_func,
        reduce_func=self._reduce_func,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class MapDataset(Dataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self,
               input_dataset,
               map_func,
               num_threads=None,
               output_buffer_size=None):
    """See `Dataset.map()` for details."""
    super(MapDataset, self).__init__()
    self._input_dataset = input_dataset

    self._output_shapes = None
    self._output_types = None

    @function.Defun(*nest.flatten(input_dataset.output_types))
    def tf_map_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      for arg, shape in zip(args, nest.flatten(input_dataset.output_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)

      if _should_unpack_args(nested_args):
        ret = map_func(*nested_args)
      else:
        ret = map_func(nested_args)

      # Extract shape information from the returned values.
      flattened_ret = [ops.convert_to_tensor(t) for t in nest.flatten(ret)]
      self._output_shapes = nest.pack_sequence_as(
          ret, [t.get_shape() for t in flattened_ret])
      self._output_types = nest.pack_sequence_as(
          ret, [t.dtype for t in flattened_ret])

      return flattened_ret

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())
    if num_threads is not None:
      self._num_threads = ops.convert_to_tensor(
          num_threads, dtype=dtypes.int32, name="num_threads")
      if output_buffer_size is not None:
        self._output_buffer_size = ops.convert_to_tensor(
            output_buffer_size, dtype=dtypes.int64, name="output_buffer_size")
      else:
        self._output_buffer_size = ops.convert_to_tensor(
            self._num_threads, dtype=dtypes.int64, name="output_buffer_size")
    else:
      self._num_threads = None
      self._output_buffer_size = None

  def make_dataset_resource(self):
    input_resource = self._input_dataset.make_dataset_resource()
    if self._num_threads is None:
      return gen_dataset_ops.map_dataset(
          input_resource,
          self._map_func.captured_inputs,
          f=self._map_func,
          output_types=nest.flatten(self.output_types),
          output_shapes=nest.flatten(self.output_shapes))
    else:
      return gen_dataset_ops.parallel_map_dataset(
          input_resource,
          self._map_func.captured_inputs,
          f=self._map_func,
          num_threads=self._num_threads,
          output_buffer_size=self._output_buffer_size,
          output_types=nest.flatten(self.output_types),
          output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class FlatMapDataset(Dataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    super(FlatMapDataset, self).__init__()
    self._input_dataset = input_dataset

    @function.Defun(*nest.flatten(input_dataset.output_types))
    def tf_map_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      for arg, shape in zip(args, nest.flatten(input_dataset.output_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)

      if _should_unpack_args(nested_args):
        dataset = map_func(*nested_args)
      else:
        dataset = map_func(nested_args)

      if not isinstance(dataset, Dataset):
        raise TypeError("`map_func` must return a `Dataset` object.")

      self._output_types = dataset.output_types
      self._output_shapes = dataset.output_shapes

      return dataset.make_dataset_resource()

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

  def make_dataset_resource(self):
    return gen_dataset_ops.flat_map_dataset(
        self._input_dataset.make_dataset_resource(),
        self._map_func.captured_inputs,
        f=self._map_func,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class FilterDataset(Dataset):
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self, input_dataset, predicate):
    """See `Dataset.filter()` for details."""
    super(FilterDataset, self).__init__()
    self._input_dataset = input_dataset

    @function.Defun(*nest.flatten(input_dataset.output_types))
    def tf_predicate(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      for arg, shape in zip(args, nest.flatten(input_dataset.output_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)

      if _should_unpack_args(nested_args):
        ret = predicate(*nested_args)
      else:
        ret = predicate(nested_args)

      ret = ops.convert_to_tensor(ret, dtype=dtypes.bool)
      if not (ret.dtype == dtypes.bool and
              ret.shape.is_compatible_with(tensor_shape.scalar())):
        raise ValueError("`predicate` must return a scalar boolean tensor.")

      return ret

    self._predicate = tf_predicate
    self._predicate.add_to_graph(ops.get_default_graph())

  def make_dataset_resource(self):
    return gen_dataset_ops.filter_dataset(
        self._input_dataset.make_dataset_resource(),
        other_arguments=self._predicate.captured_inputs,
        predicate=self._predicate,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class TextLineDataset(Dataset):
  """A `Dataset` comprising lines from one or more text files."""

  def __init__(self, filenames):
    """Creates a `TextLineDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super(TextLineDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")

  def make_dataset_resource(self):
    return gen_dataset_ops.text_line_dataset(self._filenames)

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string


class TFRecordDataset(Dataset):
  """A `Dataset` comprising records from one or more TFRecord files."""

  def __init__(self, filenames, compression_type=None):
    """Creates a `TFRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      compression_type: A `tf.string` scalar evaluating to one of `""` (no
        compression), `"ZLIB"`, or `"GZIP"`.
    """
    super(TFRecordDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(filenames, name="filenames")
    if compression_type is not None:
      self._compression_type = ops.convert_to_tensor(
          compression_type, dtype=dtypes.string, name="compression_type")
    else:
      self._compression_type = constant_op.constant("", name="compression_type")

  def make_dataset_resource(self):
    return gen_dataset_ops.tf_record_dataset(self._filenames,
                                             self._compression_type)

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([])

  @property
  def output_types(self):
    return dtypes.string


class FixedLengthRecordDataset(Dataset):
  """A `Dataset` of fixed-length records from one or more binary files."""

  def __init__(self,
               filenames,
               record_bytes,
               header_bytes=None,
               footer_bytes=None):
    """Creates a `FixedLengthRecordDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_bytes: A `tf.int64` scalar representing the number of bytes in
        each record.
      header_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to skip at the start of a file.
      footer_bytes: (Optional.) A `tf.int64` scalar representing the number of
        bytes to ignore at the end of a file.
    """
    super(FixedLengthRecordDataset, self).__init__()
    self._filenames = ops.convert_to_tensor(
        filenames, dtype=dtypes.string, name="filenames")
    self._record_bytes = ops.convert_to_tensor(
        record_bytes, dtype=dtypes.int64, name="record_bytes")
    if header_bytes is not None:
      self._header_bytes = ops.convert_to_tensor(
          header_bytes, dtype=dtypes.int64, name="header_bytes")
    else:
      self._header_bytes = constant_op.constant(
          0, dtype=dtypes.int64, name="header_bytes")
    if footer_bytes is not None:
      self._footer_bytes = ops.convert_to_tensor(
          footer_bytes, dtype=dtypes.int64, name="footer_bytes")
    else:
      self._footer_bytes = constant_op.constant(
          0, dtype=dtypes.int64, name="footer_bytes")

  def make_dataset_resource(self):
    return gen_dataset_ops.fixed_length_record_dataset(
        self._filenames, self._header_bytes, self._record_bytes,
        self._footer_bytes)

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string


def rejection_resample(dataset,
                       class_func,
                       target_dist,
                       initial_dist=None,
                       seed=None):
  """Resamples this dataset to achieve a target class distribution.

  **NOTE** Resampling is performed via rejection sampling; some fraction
  of the input values will be dropped.

  Args:
    dataset: A `Dataset` object.
    class_func: A function mapping a nested structure of tensors (having
      shapes and types defined by `dataset.output_shapes` and
      `dataset.output_types`) to a scalar `tf.int32` tensor.  Values should
      be in `[0, num_classes)`.
    target_dist: A floating point type tensor, shaped `[num_classes].
    initial_dist: (Optional.)  A floating point type tensor, shaped
      `[num_classes]`.  If not provided, the true class distribution is
      estimated live in a streaming fashion.
    seed: (Optional.) Python integer seed for the resampler.

  Returns:
    A `Dataset`.
  """
  dist_estimation_batch_size = 32
  target_dist = ops.convert_to_tensor(target_dist, name="initial_dist")
  class_values_ds = dataset.map(class_func)
  if initial_dist is not None:
    initial_dist = ops.convert_to_tensor(initial_dist, name="initial_dist")
    acceptance_dist = _calculate_acceptance_probs(initial_dist, target_dist)
    initial_dist_ds = Dataset.from_tensors(initial_dist).repeat()
    acceptance_dist_ds = Dataset.from_tensors(acceptance_dist).repeat()
  else:
    num_classes = (target_dist.shape[0].value or
                   array_ops.shape(target_dist)[0])
    smoothing_constant = 10
    num_examples_per_class_seen = resource_variable_ops.ResourceVariable(
        initial_value=array_ops.fill([num_classes],
                                     np.int64(smoothing_constant)),
        trainable=False,
        name="class_count",
        dtype=dtypes.int64)

    def update_estimate_and_tile(c):
      return array_ops.tile(
          array_ops.expand_dims(
              _estimate_data_distribution(c, num_examples_per_class_seen), 0),
          [dist_estimation_batch_size, 1])

    initial_dist_ds = (class_values_ds.batch(dist_estimation_batch_size)
                       .map(update_estimate_and_tile).unbatch())
    acceptance_dist_ds = initial_dist_ds.map(
        lambda initial: _calculate_acceptance_probs(initial, target_dist))

  def maybe_warn_on_large_rejection(accept_dist, initial_dist):
    proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
    return control_flow_ops.cond(
        math_ops.less(proportion_rejected, .5),
        lambda: accept_dist,
        lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
            accept_dist, [proportion_rejected, initial_dist, accept_dist],
            message="Proportion of examples rejected by sampler is high: ",
            summarize=100,
            first_n=10))

  acceptance_dist_ds = (Dataset.zip((acceptance_dist_ds, initial_dist_ds))
                        .map(maybe_warn_on_large_rejection))

  current_probabilities_ds = (Dataset.zip((acceptance_dist_ds, class_values_ds))
                              .map(array_ops.gather))
  filtered_ds = (
      Dataset.zip((class_values_ds, current_probabilities_ds, dataset))
      .filter(lambda _1, p, _2: random_ops.random_uniform([], seed=seed) < p))
  return filtered_ds.map(lambda class_value, _, data: (class_value, data))


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
