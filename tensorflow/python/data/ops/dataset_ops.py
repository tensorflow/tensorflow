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
"""Python wrappers for Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import threading

import numpy as np

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops


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
  def _as_variant_tensor(self):
    """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
    raise NotImplementedError("Dataset._as_variant_tensor")

  def make_initializable_iterator(self, shared_name=None):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    # ...
    sess.run(iterator.initializer)
    ```

    Args:
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the
        same devices (e.g. when using a remote server).

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          "dataset.make_initializable_iterator is not supported when eager "
          "execution is enabled.")
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator(
        container="",
        shared_name=shared_name,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))
    with ops.colocate_with(iterator_resource):
      initializer = gen_dataset_ops.make_iterator(
          self._as_variant_tensor(), iterator_resource)
    return iterator_ops.Iterator(iterator_resource, initializer,
                                 self.output_types, self.output_shapes)

  def make_one_shot_iterator(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    **N.B.** The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          "dataset.make_one_shot_iterator is not supported when eager "
          "execution is enabled.")
    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(capture_by_value=True)
    def _make_dataset():
      return self._as_variant_tensor()  # pylint: disable=protected-access

    _make_dataset.add_to_graph(ops.get_default_graph())

    return iterator_ops.Iterator(
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

  class _GeneratorState(object):
    """Stores outstanding iterators created from a Python generator.

    This class keeps track of potentially multiple iterators that may have
    been created from a generator, e.g. in the case that the dataset is
    repeated, or nested within a parallel computation.
    """

    def __init__(self, generator):
      self._generator = generator
      self._lock = threading.Lock()
      self._next_id = 0  # GUARDED_BY(self._lock)
      self._iterators = collections.defaultdict(lambda: iter(generator()))

    def get_next_id(self):
      with self._lock:
        ret = self._next_id
        self._next_id += 1
      # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
      # casting in `py_func()` will create an array of `np.int32` on Windows,
      # leading to a runtime error.
      return np.array(ret, dtype=np.int64)

    def get_iterator(self, iterator_id):
      return self._iterators[iterator_id]

    def iterator_completed(self, iterator_id):
      del self._iterators[iterator_id]

  @staticmethod
  def from_generator(generator, output_types, output_shapes=None):
    """Creates a `Dataset` whose elements are generated by `generator`.

    The `generator` argument must be a callable object that returns
    an object that support the `iter()` protocol (e.g. a generator function).
    The elements generated by `generator` must be compatible with the given
    `output_types` and (optional) `output_shapes` arguments.

    For example:

    ```python
    import itertools

    def gen():
      for i in itertools.count(1):
        yield (i, [1] * i)

    ds = Dataset.from_generator(
        gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
    value = ds.make_one_shot_iterator().get_next()

    sess.run(value)  # (1, array([1]))
    sess.run(value)  # (2, array([1, 1]))
    ```

    Args:
      generator: A callable object that takes no arguments and returns an
        object that supports the `iter()` protocol.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element yielded by `generator`.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape`
        objects corresponding to each component of an element yielded by
        `generator`.

    Returns:
      A `Dataset`.
    """
    if not callable(generator):
      raise TypeError("`generator` must be callable.")
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)

    flattened_types = nest.flatten(output_types)
    flattened_shapes = nest.flatten(output_shapes)

    generator_state = Dataset._GeneratorState(generator)

    def get_iterator_id_map_fn(unused_dummy):
      """Creates a unique `iterator_id` for each pass over the dataset.

      The "iterator_id" disambiguates between multiple concurrently
      existing iterators.

      Args:
        unused_dummy: Ignored value.

      Returns:
        A `tf.int64` tensor whose value uniquely identifies an iterator in
        `generator_state`.
      """
      return script_ops.py_func(
          generator_state.get_next_id, [], dtypes.int64, stateful=True)

    def generator_map_fn(iterator_id_t):
      """Generates the next element from iterator with ID `iterator_id_t`.

      We map this function across an infinite repetition of the
      `iterator_id_t`, and raise `StopIteration` to terminate the iteration.

      Args:
        iterator_id_t: A `tf.int64` tensor whose value uniquely identifies
          the iterator in `generator_state` from which to generate an element.

      Returns:
        A nested structure of tensors representing an element from the iterator.
      """

      def generator_py_func(iterator_id):
        """A `py_func` that will be called to invoke the iterator."""
        try:
          values = next(generator_state.get_iterator(iterator_id))
        except StopIteration:
          generator_state.iterator_completed(iterator_id)
          raise StopIteration("Iteration finished.")

        # Use the same _convert function from the py_func() implementation to
        # convert the returned values to arrays early, so that we can inspect
        # their values.
        # pylint: disable=protected-access
        ret_arrays = [
            script_ops.FuncRegistry._convert(ret, dtype=dtype.as_numpy_dtype)
            for ret, dtype in zip(nest.flatten_up_to(output_types, values),
                                  flattened_types)
        ]
        # pylint: enable=protected-access

        # Additional type and shape checking to ensure that the components
        # of the generated element match the `output_types` and `output_shapes`
        # arguments.
        for (ret_array, expected_dtype, expected_shape) in zip(
            ret_arrays, flattened_types, flattened_shapes):
          if ret_array.dtype != expected_dtype.as_numpy_dtype:
            raise TypeError(
                "`generator` yielded an element of type %s where an element "
                "of type %s was expected." % (ret_array.dtype,
                                              expected_dtype.as_numpy_dtype))
          if not expected_shape.is_compatible_with(ret_array.shape):
            raise ValueError(
                "`generator` yielded an element of shape %s where an element "
                "of shape %s was expected." % (ret_array.shape, expected_shape))

        return ret_arrays

      flat_values = script_ops.py_func(
          generator_py_func, [iterator_id_t], flattened_types, stateful=True)

      # The `py_func()` op drops the inferred shapes, so we add them back in
      # here.
      if output_shapes is not None:
        for ret_t, shape in zip(flat_values, flattened_shapes):
          ret_t.set_shape(shape)

      return nest.pack_sequence_as(output_types, flat_values)

    # This function associates each traversal of `generator` with a unique
    # iterator ID.
    def flat_map_fn(iterator_id_t):
      # First, generate an infinite dataset containing the iterator ID repeated
      # forever.
      repeated_id = Dataset.from_tensors(iterator_id_t).repeat(None)

      # The `generator_map_fn` gets the next element from the iterator with the
      # relevant ID, and raises StopIteration when that iterator contains no
      # more elements.
      return repeated_id.map(generator_map_fn)

    # A single-element dataset that, each time it is evaluated, contains a
    # freshly-generated and unique (for the returned dataset) int64
    # ID that will be used to identify the appropriate Python state, which
    # is encapsulated in `generator_state`, and captured in
    # `get_iterator_id_map_fn`.
    dummy = 0
    id_dataset = Dataset.from_tensors(dummy).map(get_iterator_id_map_fn)

    # A dataset that contains all of the elements generated by a
    # single iterator created from `generator`, identified by the
    # iterator ID contained in `id_dataset`. Lifting the iteration
    # into a flat_map here enables multiple repetitions and/or nested
    # versions of the returned dataset to be created, because it forces
    # the generation of a new ID for each version.
    return id_dataset.flat_map(flat_map_fn)

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

  def concatenate(self, dataset):
    """Creates a `Dataset` by concatenating given dataset with this dataset.

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3 }
    b = { 4, 5, 6, 7 }

    # Input dataset and dataset to be concatenated should have same
    # nested structures and output types.
    # c = { (8, 9), (10, 11), (12, 13) }
    # d = { 14.0, 15.0, 16.0 }
    # a.concatenate(c) and a.concatenate(d) would result in error.

    a.concatenate(b) == { 1, 2, 3, 4, 5, 6, 7 }
    ```

    Args:
      dataset: `Dataset` to be concatenated.

    Returns:
      A `Dataset`.
    """
    return ConcatenateDataset(self, dataset)

  def prefetch(self, buffer_size):
    """Creates a `Dataset` that prefetches elements from this dataset.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        maximum number elements that will be buffered when prefetching.

    Returns:
      A `Dataset`.
    """
    return PrefetchDataset(self, buffer_size)

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

  def _enumerate(self, start=0):

    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    return Dataset.zip((Dataset.range(start, max_value), self))

  def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        random seed that will be used to create the distribution. See
        @{tf.set_random_seed} for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      A `Dataset`.
    """
    return ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)

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

  def shard(self, num_shards, index):
    """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.

    When reading a single input file, you can skip elements as follows:

    ```python
    d = tf.data.TFRecordDataset(FLAGS.input_file)
    d = d.shard(FLAGS.num_workers, FLAGS.worker_index)
    d = d.repeat(FLAGS.num_epochs)
    d = d.shuffle(FLAGS.shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=FLAGS.num_map_threads)
    ```

    Important caveats:

    - Be sure to shard before you use any randomizing operator (such as
      shuffle).
    - Generally it is best if the shard operator is used early in the dataset
      pipeline. For example, when reading from a set of TFRecord files, shard
      before converting the dataset to input samples. This avoids reading every
      file on every worker. The following is an example of an efficient
      sharding strategy within a complete pipeline:

    ```python
    d = Dataset.list_files(FLAGS.pattern)
    d = d.shard(FLAGS.num_workers, FLAGS.worker_index)
    d = d.repeat(FLAGS.num_epochs)
    d = d.shuffle(FLAGS.shuffle_buffer_size)
    d = d.repeat()
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=FLAGS.num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=FLAGS.num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if `num_shards` or `index` are illegal values. Note: error
        checking is done on a best-effort basis, and aren't guaranteed to be
        caught upon dataset creation. (e.g. providing in a placeholder tensor
        bypasses the early checking, and will instead result in an error during
        a session.run call.)
    """
    num_shards = ops.convert_to_tensor(
        num_shards, name="num_shards", dtype=dtypes.int64)
    num_shards_static = tensor_util.constant_value(num_shards)
    index = ops.convert_to_tensor(index, name="index", dtype=dtypes.int64)
    index_static = tensor_util.constant_value(index)

    if num_shards_static is not None and num_shards_static < 1:
      raise ValueError("num_shards must be >= 1; got: %s" % num_shards_static)
    if index_static is not None and index_static < 0:
      raise ValueError("index must be >= 0; got: %s" % index_static)
    if (index_static is not None and num_shards_static is not None and
        index_static >= num_shards_static):
      raise ValueError("index must be <= num_shards; %s is not < %s" %
                       (index_static, num_shards_static))

    def filter_fn(elem_index, _):
      mod_result = math_ops.mod(elem_index, num_shards)
      return math_ops.equal(mod_result, index)

    return self._enumerate().filter(filter_fn).map(lambda _, elem: elem)

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

  def map(self, map_func, num_parallel_calls=None):
    """Maps `map_func` across this datset.

    Args:
      map_func: A function mapping a nested structure of tensors (having
        shapes and types defined by `self.output_shapes` and
       `self.output_types`) to another nested structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process in parallel. If not
        specified, elements will be processed sequentially.

    Returns:
      A `Dataset`.
    """
    if num_parallel_calls is None:
      return MapDataset(self, map_func)
    else:
      return ParallelMapDataset(self, map_func, num_parallel_calls)

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

  def interleave(self, map_func, cycle_length, block_length=1):
    """Maps `map_func` across this dataset, and interleaves the results.

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    ```python
    # Preprocess 4 files concurrently, and interleave blocks of 16 records from
    # each file.
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt", ..."]
    dataset = (Dataset.from_tensor_slices(filenames)
               .interleave(lambda x:
                   TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                   cycle_length=4, block_length=16))
    ```

    The `cycle_length` and `block_length` arguments control the order in which
    elements are produced. `cycle_length` controls the number of input elements
    that are processed concurrently. If you set `cycle_length` to 1, this
    transformation will handle one input element at a time, and will produce
    identical results = to @{tf.data.Dataset.flat_map}. In general,
    this transformation will apply `map_func` to `cycle_length` input elements,
    open iterators on the returned `Dataset` objects, and cycle through them
    producing `block_length` consecutive elements from each iterator, and
    consuming the next input element each time it reaches the end of an
    iterator.

    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3, 4, 5 }

    # NOTE: New lines indicate "block" boundaries.
    a.interleave(lambda x: Dataset.from_tensors(x).repeat(6),
                 cycle_length=2, block_length=4) == {
        1, 1, 1, 1,
        2, 2, 2, 2,
        1, 1,
        2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        3, 3,
        4, 4,
        5, 5, 5, 5,
        5, 5,
    }
    ```

    NOTE: The order of elements yielded by this transformation is
    deterministic, as long as `map_func` is a pure function. If
    `map_func` contains any stateful operations, the order in which
    that state is accessed is undefined.

    Args:
      map_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        `Dataset`.
      cycle_length: The number of elements from this dataset that will be
        processed concurrently.
      block_length: The number of consecutive elements to produce from each
        input element before cycling to another input element.

    Returns:
      A `Dataset`.
    """
    return InterleaveDataset(self, map_func, cycle_length, block_length)

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

  def apply(self, transformation_func):
    """Apply a transformation function to this dataset.

    `apply` enables chaining of custom `Dataset` transformations, which are
    represented as functions that take one `Dataset` argument and return a
    transformed `Dataset`.

    For example:

    ```
    dataset = (dataset.map(lambda x: x ** 2)
               .apply(group_by_window(key_func, reduce_func, window_size))
               .map(lambda x: x ** 3))
    ```

    Args:
      transformation_func: A function that takes one `Dataset` argument and
        returns a `Dataset`.

    Returns:
      The `Dataset` returned by applying `transformation_func` to this dataset.
    """
    dataset = transformation_func(self)
    if not isinstance(dataset, Dataset):
      raise TypeError("`transformation_func` must return a Dataset.")
    return dataset


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

  def _as_variant_tensor(self):
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

  def _as_variant_tensor(self):
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

  def _as_variant_tensor(self):
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
    for ds in nest.flatten(datasets):
      if not isinstance(ds, Dataset):
        if isinstance(ds, list):
          message = ("The argument to `Dataset.zip()` must be a nested "
                     "structure of `Dataset` objects. Nested structures do not "
                     "support Python lists; please use a tuple instead.")
        else:
          message = ("The argument to `Dataset.zip()` must be a nested "
                     "structure of `Dataset` objects.")
        raise TypeError(message)
    self._datasets = datasets

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.zip_dataset(
        [ds._as_variant_tensor() for ds in nest.flatten(self._datasets)],
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
    # pylint: enable=protected-access

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


class ConcatenateDataset(Dataset):
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate):
    """See `Dataset.concatenate()` for details."""
    super(ConcatenateDataset, self).__init__()
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate
    nest.assert_same_structure(input_dataset.output_types,
                               dataset_to_concatenate.output_types)
    for a, b in zip(
        nest.flatten(input_dataset.output_types),
        nest.flatten(dataset_to_concatenate.output_types)):
      if a != b:
        raise TypeError(
            "Two datasets to concatenate have different types %s and %s" %
            (input_dataset.output_types, dataset_to_concatenate.output_types))

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.concatenate_dataset(
        self._input_dataset._as_variant_tensor(),
        self._dataset_to_concatenate._as_variant_tensor(),
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))
    # pylint: enable=protected-access

  @property
  def output_shapes(self):
    return nest.pack_sequence_as(self._input_dataset.output_shapes, [
        ts1.most_specific_compatible_shape(ts2)
        for (ts1, ts2) in zip(
            nest.flatten(self._input_dataset.output_shapes),
            nest.flatten(self._dataset_to_concatenate.output_shapes))
    ])

  @property
  def output_types(self):
    return self._input_dataset.output_types


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

  def _as_variant_tensor(self):
    return gen_dataset_ops.repeat_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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
      self._stop = self._build_tensor(args[0], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 2:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 3:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(args[2], "step")
    else:
      raise ValueError("Invalid arguments to RangeDataset: %s" % str(args))

  def _build_tensor(self, int64_value, name):
    return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

  def _as_variant_tensor(self):
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

  def _as_variant_tensor(self):
    return gen_dataset_ops.cache_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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

  def __init__(self, input_dataset, buffer_size, seed=None,
               reshuffle_each_iteration=None):
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
    if reshuffle_each_iteration is None:
      self._reshuffle_each_iteration = True
    else:
      self._reshuffle_each_iteration = reshuffle_each_iteration

  def _as_variant_tensor(self):
    return gen_dataset_ops.shuffle_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        seed=self._seed,
        seed2=self._seed2,
        reshuffle_each_iteration=self._reshuffle_each_iteration,
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

  def _as_variant_tensor(self):
    return gen_dataset_ops.take_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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

  def _as_variant_tensor(self):
    return gen_dataset_ops.skip_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        count=self._count,
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
    self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64,
                                             name="batch_size")

  def _as_variant_tensor(self):
    return gen_dataset_ops.batch_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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
    self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64,
                                             name="batch_size")
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

  def _as_variant_tensor(self):
    return gen_dataset_ops.padded_batch_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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


def _should_unpack_args(args):
  """Returns `True` if `args` should be `*args` when passed to a callable."""
  return type(args) is tuple  # pylint: disable=unidiomatic-typecheck


class MapDataset(Dataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self, input_dataset, map_func):
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

      # If `map_func` returns a list of tensors, `nest.flatten()` and
      # `ops.convert_to_tensor()` would conspire to attempt to stack
      # those tensors into a single tensor, because the customized
      # version of `nest.flatten()` does not recurse into lists. Since
      # it is more likely that the list arose from returning the
      # result of an operation (such as `tf.py_func()`) that returns a
      # list of not-necessarily-stackable tensors, we treat the
      # returned value is a `tuple` instead. A user wishing to pack
      # the return value into a single tensor can use an explicit
      # `tf.stack()` before returning.
      if isinstance(ret, list):
        ret = tuple(ret)

      # Extract shape information from the returned values.
      flattened_ret = [ops.convert_to_tensor(t) for t in nest.flatten(ret)]
      self._output_shapes = nest.pack_sequence_as(
          ret, [t.get_shape() for t in flattened_ret])
      self._output_types = nest.pack_sequence_as(
          ret, [t.dtype for t in flattened_ret])

      return flattened_ret

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    return gen_dataset_ops.map_dataset(
        input_t,
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


class ParallelMapDataset(MapDataset):
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self, input_dataset, map_func, num_parallel_calls):
    """See `Dataset.map()` for details."""
    super(ParallelMapDataset, self).__init__(input_dataset, map_func)

    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int32, name="num_parallel_calls")

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    # pylint: disable=protected-access
    return gen_dataset_ops.parallel_map_dataset(
        input_t,
        self._map_func.captured_inputs,
        f=self._map_func,
        num_parallel_calls=self._num_parallel_calls,
        output_types=nest.flatten(self.output_types),
        output_shapes=nest.flatten(self.output_shapes))
    # pylint: enable=protected-access


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

      return dataset._as_variant_tensor()  # pylint: disable=protected-access

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

  def _as_variant_tensor(self):
    return gen_dataset_ops.flat_map_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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


class InterleaveDataset(Dataset):
  """A `Dataset` that maps a function over its input and interleaves the result.
  """

  def __init__(self, input_dataset, map_func, cycle_length, block_length):
    """See `Dataset.interleave()` for details."""
    super(InterleaveDataset, self).__init__()
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

      return dataset._as_variant_tensor()  # pylint: disable=protected-access

    self._map_func = tf_map_func
    self._map_func.add_to_graph(ops.get_default_graph())

    self._cycle_length = ops.convert_to_tensor(cycle_length, dtype=dtypes.int64,
                                               name="cycle_length")
    self._block_length = ops.convert_to_tensor(block_length, dtype=dtypes.int64,
                                               name="block_length")

  def _as_variant_tensor(self):
    return gen_dataset_ops.interleave_dataset(
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

  def _as_variant_tensor(self):
    return gen_dataset_ops.filter_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
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


class PrefetchDataset(Dataset):
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size):
    """See `Dataset.prefetch()` for details."""
    super(PrefetchDataset, self).__init__()
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(buffer_size, dtype=dtypes.int64,
                                              name="buffer_size")

  def _as_variant_tensor(self):
    return gen_dataset_ops.prefetch_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        output_shapes=nest.flatten(self.output_shapes),
        output_types=nest.flatten(self.output_types))

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types
