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
import threading
import warnings

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import sparse
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.Dataset")
class Dataset(object):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements (nested structures of tensors) and a "logical
  plan" of transformations that act on those elements.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  def _as_serialized_graph(self):
    """Produces serialized graph representation of the dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a
      serialized graph.
    """
    return gen_dataset_ops.dataset_to_graph(self._as_variant_tensor())

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
    if context.executing_eagerly():
      raise RuntimeError(
          "dataset.make_initializable_iterator is not supported when eager "
          "execution is enabled.")
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator(
        container="", shared_name=shared_name, **flat_structure(self))
    with ops.colocate_with(iterator_resource):
      initializer = gen_dataset_ops.make_iterator(self._as_variant_tensor(),
                                                  iterator_resource)
    return iterator_ops.Iterator(iterator_resource, initializer,
                                 self.output_types, self.output_shapes,
                                 self.output_classes)

  def __iter__(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    The returned iterator implements the Python iterator protocol and therefore
    can only be used in eager mode.

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If eager execution is not enabled.
    """
    if context.executing_eagerly():
      return iterator_ops.EagerIterator(self)
    else:
      raise RuntimeError("dataset.__iter__() is only supported when eager "
                         "execution is enabled.")

  def make_one_shot_iterator(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.
    """
    if context.executing_eagerly():
      return iterator_ops.EagerIterator(self)
    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(capture_by_value=True)
    def _make_dataset():
      return self._as_variant_tensor()  # pylint: disable=protected-access

    try:
      _make_dataset.add_to_graph(ops.get_default_graph())
    except ValueError as err:
      capture_err = "Cannot capture a stateful node"
      if capture_err in str(err):
        raise ValueError(
            "Failed to create a one-shot iterator for a dataset. "
            "`Dataset.make_one_shot_iterator()` does not support datasets that "
            "capture stateful objects, such as a `Variable` or `LookupTable`. "
            "In these cases, use `Dataset.make_initializable_iterator()`. "
            "(Original error: %s)" % err)
      else:
        six.reraise(ValueError, err)

    return iterator_ops.Iterator(
        gen_dataset_ops.one_shot_iterator(
            dataset_factory=_make_dataset, **flat_structure(self)),
        None, self.output_types, self.output_shapes, self.output_classes)

  @abc.abstractproperty
  def output_classes(self):
    """Returns the class of each component of an element of this dataset.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    raise NotImplementedError("Dataset.output_classes")

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

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    @{tf.constant} operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization.  If tensors contains
    one or more large NumPy arrays, consider the alternative described in
    @{$programmers_guide/datasets#consuming_numpy_arrays$this guide}.

    Args:
      tensors: A nested structure of tensors.

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorDataset(tensors)

  @staticmethod
  def from_tensor_slices(tensors):
    """Creates a `Dataset` whose elements are slices of the given tensors.

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    @{tf.constant} operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization.  If tensors contains
    one or more large NumPy arrays, consider the alternative described in
    @{$programmers_guide/datasets#consuming_numpy_arrays$this guide}.

    Args:
      tensors: A nested structure of tensors, each having the same size in the
        0th dimension.

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorSliceDataset(tensors)

  @staticmethod
  @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
  def from_sparse_tensor_slices(sparse_tensor):
    """Splits each rank-N `tf.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
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
      self._args = {}
      self._iterators = {}

    def get_next_id(self, *args):
      with self._lock:
        ret = self._next_id
        self._next_id += 1
      self._args[ret] = args
      # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
      # casting in `py_func()` will create an array of `np.int32` on Windows,
      # leading to a runtime error.
      return np.array(ret, dtype=np.int64)

    def get_iterator(self, iterator_id):
      try:
        return self._iterators[iterator_id]
      except KeyError:
        iterator = iter(self._generator(*self._args.pop(iterator_id)))
        self._iterators[iterator_id] = iterator
        return iterator

    def iterator_completed(self, iterator_id):
      del self._iterators[iterator_id]

  @staticmethod
  def from_generator(generator, output_types, output_shapes=None, args=None):
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

    NOTE: The current implementation of `Dataset.from_generator()` uses
    @{tf.py_func} and inherits the same constraints. In particular, it
    requires the `Dataset`- and `Iterator`-related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. The body of `generator` will not be
    serialized in a `GraphDef`, and you should not use this method if you
    need to serialize your model and restore it in a different environment.

    NOTE: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.

    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take
        no arguments; otherwise it must take as many arguments as there are
        values in `args`.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element yielded by `generator`.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape`
        objects corresponding to each component of an element yielded by
        `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.

    Returns:
      Dataset: A `Dataset`.
    """
    if not callable(generator):
      raise TypeError("`generator` must be callable.")
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)
    if args is None:
      args = ()
    else:
      args = tuple(ops.convert_n_to_tensor(args, name="args"))

    flattened_types = [dtypes.as_dtype(dt) for dt in nest.flatten(output_types)]
    flattened_shapes = nest.flatten(output_shapes)

    generator_state = Dataset._GeneratorState(generator)

    def get_iterator_id_fn(unused_dummy):
      """Creates a unique `iterator_id` for each pass over the dataset.

      The returned `iterator_id` disambiguates between multiple concurrently
      existing iterators.

      Args:
        unused_dummy: Ignored value.

      Returns:
        A `tf.int64` tensor whose value uniquely identifies an iterator in
        `generator_state`.
      """
      return script_ops.py_func(
          generator_state.get_next_id, args, dtypes.int64, stateful=True)

    def generator_next_fn(iterator_id_t):
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
        # `next()` raises `StopIteration` when there are no more
        # elements remaining to be generated.
        values = next(generator_state.get_iterator(iterator_id))

        # Use the same _convert function from the py_func() implementation to
        # convert the returned values to arrays early, so that we can inspect
        # their values.
        try:
          flattened_values = nest.flatten_up_to(output_types, values)
        except (TypeError, ValueError):
          raise TypeError(
              "`generator` yielded an element that did not match the expected "
              "structure. The expected structure was %s, but the yielded "
              "element was %s." % (output_types, values))
        ret_arrays = []
        for ret, dtype in zip(flattened_values, flattened_types):
          try:
            ret_arrays.append(script_ops.FuncRegistry._convert(  # pylint: disable=protected-access
                ret, dtype=dtype.as_numpy_dtype))
          except (TypeError, ValueError):
            raise TypeError(
                "`generator` yielded an element that could not be converted to "
                "the expected type. The expected type was %s, but the yielded "
                "element was %s." % (dtype.name, ret))

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

    def finalize_fn(iterator_id_t):
      """Releases host-side state for the iterator with ID `iterator_id_t`."""

      def finalize_py_func(iterator_id):
        generator_state.iterator_completed(iterator_id)
        # We return a dummy value so that the `finalize_fn` has a valid
        # signature.
        # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
        # casting in `py_func()` will create an array of `np.int32` on Windows,
        # leading to a runtime error.
        return np.array(0, dtype=np.int64)

      return script_ops.py_func(
          finalize_py_func, [iterator_id_t], dtypes.int64, stateful=True)

    # This function associates each traversal of `generator` with a unique
    # iterator ID.
    def flat_map_fn(dummy_arg):
      # The `get_iterator_id_fn` gets a unique ID for the current instance of
      # of the generator.
      # The `generator_next_fn` gets the next element from the iterator with the
      # given ID, and raises StopIteration when that iterator contains no
      # more elements.
      return _GeneratorDataset(dummy_arg, get_iterator_id_fn, generator_next_fn,
                               finalize_fn)

    # A single-element dataset that, each time it is evaluated, contains a
    # freshly-generated and unique (for the returned dataset) int64
    # ID that will be used to identify the appropriate Python state, which
    # is encapsulated in `generator_state`, and captured in
    # `get_iterator_id_map_fn`.
    dummy = 0
    id_dataset = Dataset.from_tensors(dummy)

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
      Dataset: A `RangeDataset`.

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
      Dataset: A `Dataset`.
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
      Dataset: A `Dataset`.
    """
    return ConcatenateDataset(self, dataset)

  def prefetch(self, buffer_size):
    """Creates a `Dataset` that prefetches elements from this dataset.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        maximum number of elements that will be buffered when prefetching.

    Returns:
      Dataset: A `Dataset`.
    """
    return PrefetchDataset(self, buffer_size)

  @staticmethod
  def list_files(file_pattern, shuffle=None, seed=None):
    """A dataset of all files matching a pattern.

    NOTE: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.

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
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        random seed that will be used to create the distribution. See
        @{tf.set_random_seed} for behavior.

    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    """
    if shuffle is None:
      shuffle = True
    matching_files = gen_io_ops.matching_files(file_pattern)
    dataset = Dataset.from_tensor_slices(matching_files)
    if shuffle:
      # NOTE(mrry): The shuffle buffer size must be greater than zero, but the
      # list of files might be empty.
      buffer_size = math_ops.maximum(
          array_ops.shape(matching_files, out_type=dtypes.int64)[0], 1)
      dataset = dataset.shuffle(buffer_size, seed=seed)
    return dataset

  def repeat(self, count=None):
    """Repeats this dataset `count` times.

    NOTE: If this dataset is a function of global state (e.g. a random number
    generator), then different repetitions may produce different elements.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior
        (if `count` is `None` or `-1`) is for the dataset be repeated
        indefinitely.

    Returns:
      Dataset: A `Dataset`.
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
      Dataset: A `Dataset`.
    """
    return ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)

  def cache(self, filename=""):
    """Caches the elements in this dataset.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching tensors in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.

    Returns:
      Dataset: A `Dataset`.
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
      Dataset: A `Dataset`.
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
      Dataset: A `Dataset`.
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
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=FLAGS.num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=FLAGS.num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

    Returns:
      Dataset: A `Dataset`.

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

  def batch(self, batch_size, drop_remainder=False):
    """Combines consecutive elements of this dataset into batches.

    The tensors in the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case its has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    """
    return BatchDataset(self, batch_size, drop_remainder)

  def padded_batch(self,
                   batch_size,
                   padded_shapes,
                   padding_values=None,
                   drop_remainder=False):
    """Combines consecutive elements of this dataset into padded batches.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element.

    Like @{tf.data.Dataset.batch}, the tensors in the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike @{tf.data.Dataset.batch}, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padding_shapes`. The `padding_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant (e.g. `tf.Dimension(37)`), the component
      will be padded out to that length in that dimension.
    * If the dimension is unknown (e.g. `tf.Dimension(None)`), the component
      will be padded out to the maximum length of all elements in that
      dimension.

    See also @{tf.contrib.data.dense_to_sparse_batch}, which combines elements
    that may have different shapes into a @{tf.SparseTensor}.

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
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case its has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    """
    return PaddedBatchDataset(self, batch_size, padded_shapes, padding_values,
                              drop_remainder)

  def map(self, map_func, num_parallel_calls=None):
    """Maps `map_func` across this dataset.

    Args:
      map_func: A function mapping a nested structure of tensors (having
        shapes and types defined by `self.output_shapes` and
       `self.output_types`) to another nested structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process in parallel. If not
        specified, elements will be processed sequentially.

    Returns:
      Dataset: A `Dataset`.
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
      Dataset: A `Dataset`.
    """
    return FlatMapDataset(self, map_func)

  def interleave(self, map_func, cycle_length, block_length=1):
    """Maps `map_func` across this dataset, and interleaves the results.

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    ```python
    # Preprocess 4 files concurrently, and interleave blocks of 16 records from
    # each file.
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt", ...]
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
      Dataset: A `Dataset`.
    """
    return InterleaveDataset(self, map_func, cycle_length, block_length)

  def filter(self, predicate):
    """Filters this dataset according to `predicate`.

    Args:
      predicate: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        scalar `tf.bool` tensor.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
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
      Dataset: The `Dataset` returned by applying `transformation_func` to this
          dataset.
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
      tensors = nest.pack_sequence_as(tensors, [
          sparse_tensor_lib.SparseTensor.from_value(t)
          if sparse_tensor_lib.is_sparse(t) else ops.convert_to_tensor(
              t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(tensors))
      ])

    self._tensors = sparse.serialize_sparse_tensors(tensors)
    self._output_classes = sparse.get_classes(tensors)
    self._output_shapes = nest.pack_sequence_as(
        tensors, [t.get_shape() for t in nest.flatten(tensors)])
    self._output_types = nest.pack_sequence_as(
        tensors, [t.dtype for t in nest.flatten(tensors)])

  def _as_variant_tensor(self):
    return gen_dataset_ops.tensor_dataset(
        nest.flatten(self._tensors),
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


class TensorSliceDataset(Dataset):
  """A `Dataset` of slices from a nested structure of tensors."""

  def __init__(self, tensors):
    """See `Dataset.from_tensor_slices()` for details."""
    super(TensorSliceDataset, self).__init__()
    with ops.name_scope("tensors"):
      tensors = nest.pack_sequence_as(tensors, [
          sparse_tensor_lib.SparseTensor.from_value(t)
          if sparse_tensor_lib.is_sparse(t) else ops.convert_to_tensor(
              t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(tensors))
      ])
      flat_tensors = nest.flatten(tensors)

    batch_dim = flat_tensors[0].get_shape()[0]
    for t in flat_tensors[1:]:
      batch_dim.assert_is_compatible_with(t.get_shape()[0])
    self._tensors = sparse.serialize_many_sparse_tensors(tensors)
    self._output_classes = sparse.get_classes(tensors)
    self._output_shapes = nest.pack_sequence_as(
        tensors, [t.get_shape()[1:] for t in nest.flatten(tensors)])
    self._output_types = nest.pack_sequence_as(
        tensors, [t.dtype for t in nest.flatten(tensors)])

  def _as_variant_tensor(self):
    return gen_dataset_ops.tensor_slice_dataset(
        nest.flatten(self._tensors),
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
  def output_classes(self):
    return (ops.Tensor, ops.Tensor, ops.Tensor)

  @property
  def output_shapes(self):
    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape[1] - 1).merge_with(shape_shape[0] - 1)
    num_values = tensor_shape.Dimension(None)
    return (tensor_shape.TensorShape([num_values, rank]),
            tensor_shape.TensorShape([num_values]),
            tensor_shape.TensorShape([rank]))

  @property
  def output_types(self):
    return (dtypes.int64, self._sparse_tensor.dtype, dtypes.int64)


class _NestedDatasetComponent(object):
  """The structure of a `Dataset` nested in a component of another `Dataset`.

  A `StructuredFunctionWrapper` around a function that returns a `Dataset` as
  one of its components will have a `NestedDatasetComponent` in the
  corresponding position in the `output_classes`, `output_shapes`, and
  `output_types` properties.

  NOTE(mrry): This class is not currently exposed via the public API. Support
  for nested datasets can be enabled on a function-by-function basis by setting
  `experimental_nested_dataset_support=True` in the `StructuredFunctionWrapper`
  initializer.

  TODO(b/110122868): Add this class, or something equivalent, to the public API.
  We are considering revising the public API for accessing Dataset structure
  (`output_classes` etc.) based on experience with nested datasets and other
  custom component types.
  """

  def __init__(self, dataset):
    self._output_classes = dataset.output_classes
    self._output_shapes = dataset.output_shapes
    self._output_types = dataset.output_types

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class _VariantDataset(Dataset):
  """A Dataset wrapper around a @{tf.variant}-typed function argument."""

  def __init__(self, dataset_variant, structure):
    super(_VariantDataset, self).__init__()
    self._dataset_variant = dataset_variant
    self._structure = structure

  def _as_variant_tensor(self):
    return self._dataset_variant

  @property
  def output_classes(self):
    return self._structure.output_classes

  @property
  def output_shapes(self):
    return self._structure.output_shapes

  @property
  def output_types(self):
    return self._structure.output_types


class StructuredFunctionWrapper(object):
  """A wrapper for `Defun` that supports structured arguments and return values.
  """

  def __init__(self, func, transformation_name, dataset=None,
               input_classes=None, input_shapes=None, input_types=None,
               add_to_graph=True, experimental_nested_dataset_support=False):
    """Creates a new `StructuredFunctionWrapper` for the given function.

    Args:
      func: A function from a nested structure to another nested structure.
      transformation_name: Human-readable name of the transformation in which
        this function is being instantiated, for error messages.
      dataset: (Optional.) A @{tf.data.Dataset}. If given, the structure of this
        dataset will be assumed as the structure for `func` arguments; otherwise
        `input_classes`, `input_shapes`, and `input_types` must be defined.
      input_classes: (Optional.) A nested structure of `type`. If given, this
        argument defines the Python types for `func` arguments.
      input_shapes: (Optional.) A nested structure of @{tf.TensorShape}. If
        given, this argument defines the shapes and structure for `func`
        arguments.
      input_types: (Optional.) A nested structure of @{tf.DType}. If given, this
        argument defines the element types and structure for `func` arguments.
      add_to_graph: (Optional.) If `True`, the function will be added to the
        default graph.
      experimental_nested_dataset_support: (Optional.) If `True`, the function
        will support @{tf.data.Dataset} objects as arguments and return values.

    Raises:
      ValueError: If an invalid combination of `dataset`, `input_classes`,
        `input_shapes`, and `input_types` is passed.
    """
    if dataset is None:
      if input_classes is None or input_shapes is None or input_types is None:
        raise ValueError("Either `dataset`, or all of `input_classes`, "
                         "`input_shapes`, and `input_types` must be specified.")
      self._input_shapes = input_shapes
      self._input_types = input_types
      self._input_classes = input_classes
    else:
      if not (input_classes is None and input_shapes is None and
              input_types is None):
        raise ValueError("Either `dataset`, or all of `input_classes`, "
                         "`input_shapes`, and `input_types` must be specified.")
      self._input_shapes = dataset.output_shapes
      self._input_types = dataset.output_types
      self._input_classes = dataset.output_classes

    self._transformation_name = transformation_name

    # TODO(b/110122868): Enable this support for all `tf.data` functions.
    self._nested_dataset_support = experimental_nested_dataset_support

    @function.Defun(*self._defun_args())
    def tf_data_structured_function_wrapper(*args):
      """Wrapper for passing nested structures to and from tf.data functions."""
      flat_args = []
      for arg, arg_class, arg_shape, arg_type in zip(
          args,
          nest.flatten(self._input_classes),
          nest.flatten(self._input_shapes),
          nest.flatten(self._input_types)):
        # TODO(b/110122868): Add a registration mechanism for new component
        # types.
        if arg_class is sparse_tensor_lib.SparseTensor:
          arg = sparse.deserialize_sparse_tensors(
              arg, arg_type, arg_shape, arg_class)
          arg.indices.set_shape([None, arg_shape.ndims])
          arg.dense_shape.set_shape([arg_shape.ndims])
        elif isinstance(arg_class, _NestedDatasetComponent):
          assert self._nested_dataset_support
          arg = _VariantDataset(arg, arg_class)
        else:
          arg.set_shape(arg_shape)
        flat_args.append(arg)
      nested_args = nest.pack_sequence_as(self._input_classes, flat_args)
      if not _should_unpack_args(nested_args):
        nested_args = (nested_args,)

      ret = func(*nested_args)
      # If `func` returns a list of tensors, `nest.flatten()` and
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

      # Convert any `SparseTensorValue`s to `SparseTensor`s and all other
      # values to tensors.
      flat_ret = []
      flat_classes = []
      flat_shapes = []
      flat_types = []
      for t in nest.flatten(ret):
        # TODO(b/110122868): Add a registration mechanism for new component
        # types.
        if sparse_tensor_lib.is_sparse(t):
          t = sparse_tensor_lib.SparseTensor.from_value(t)
          flat_ret.append(sparse.serialize_sparse_tensors(t))
          flat_classes.append(sparse_tensor_lib.SparseTensor)
          flat_shapes.append(t.get_shape())
          flat_types.append(t.dtype)
        elif isinstance(t, Dataset):
          if not self._nested_dataset_support:
            raise NotImplementedError(
                "The %s transformation does not currently support nested "
                "datasets as outputs." % self._transformation_name)

          flat_ret.append(t._as_variant_tensor())  # pylint: disable=protected-access
          component = _NestedDatasetComponent(t)
          flat_classes.append(component)
          flat_shapes.append(component)
          flat_types.append(component)
        else:
          t = ops.convert_to_tensor(t)
          flat_ret.append(t)
          flat_classes.append(ops.Tensor)
          flat_shapes.append(t.get_shape())
          flat_types.append(t.dtype)

      ret = nest.pack_sequence_as(ret, flat_ret)
      self._output_classes = nest.pack_sequence_as(ret, flat_classes)
      self._output_shapes = nest.pack_sequence_as(ret, flat_shapes)
      self._output_types = nest.pack_sequence_as(ret, flat_types)

      _warn_if_collections(transformation_name)

      return flat_ret

    self._function = tf_data_structured_function_wrapper
    if add_to_graph:
      self._function.add_to_graph(ops.get_default_graph())
    else:
      # Use the private method that will execute
      # `tf_data_structured_function_wrapper` but delay adding it to the graph
      # in case (e.g.) we need to rerun the function.
      self._function._create_definition_if_needed()  # pylint: disable=protected-access

  def _defun_args(self):
    """Returns a flat list of @{tf.DType} for the input element structure."""
    ret = []
    for input_type, input_class in zip(nest.flatten(self._input_types),
                                       nest.flatten(self._input_classes)):
      # TODO(b/110122868): Add a registration mechanism for new component types.
      if input_class is sparse_tensor_lib.SparseTensor:
        ret.append(dtypes.variant)
      elif isinstance(input_class, _NestedDatasetComponent):
        if not self._nested_dataset_support:
          raise NotImplementedError(
              "The %s transformation does not currently support nested "
              "datasets as inputs." % self._transformation_name)
        ret.append(dtypes.variant)
      else:
        assert isinstance(input_type, dtypes.DType)
        ret.append(input_type)
    return ret

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types

  @property
  def function(self):
    return self._function


def flat_structure(dataset):
  """Helper for setting `output_shapes` and `output_types` attrs of Dataset ops.

  Most Dataset op constructors expect `output_shapes` and `output_types`
  arguments that represent the flattened structure of an element. This helper
  function generates these attrs as a keyword argument dictionary, allowing
  `Dataset._as_variant_tensor()` implementations to pass
  `**flat_structure(self)` to the op constructor.

  Args:
    dataset: A @{tf.data.Dataset}.

  Returns:
    A dictionary of keyword arguments that can be passed to many Dataset op
    constructors.
  """
  return {
      "output_shapes": nest.flatten(sparse.as_dense_shapes(
          dataset.output_shapes, dataset.output_classes)),
      "output_types": nest.flatten(sparse.as_dense_types(
          dataset.output_types, dataset.output_classes)),
  }


class _GeneratorDataset(Dataset):
  """A `Dataset` that generates elements by invoking a function."""

  def __init__(self, init_args, init_func, next_func, finalize_func):
    """Constructs a `_GeneratorDataset`.

    Args:
      init_args: A nested structure representing the arguments to `init_func`.
      init_func: A TensorFlow function that will be called on `init_args` each
        time a C++ iterator over this dataset is constructed. Returns a nested
        structure representing the "state" of the dataset.
      next_func: A TensorFlow function that will be called on the result of
        `init_func` to produce each element, and that raises `OutOfRangeError`
        to terminate iteration.
      finalize_func: A TensorFlow function that will be called on the result of
        `init_func` immediately before a C++ iterator over this dataset is
        destroyed. The return value is ignored.
    """
    super(_GeneratorDataset, self).__init__()
    # These members will be initialized by `tf_init_func`.
    self._state_classes = None
    self._state_shapes = None
    self._state_types = None

    self._init_args = init_args

    init_args_classes = sparse.get_classes(init_args)
    init_args_shapes = nest.pack_sequence_as(
        init_args, [t.get_shape() for t in nest.flatten(init_args)])
    init_args_types = nest.pack_sequence_as(
        init_args, [t.dtype for t in nest.flatten(init_args)])

    wrapped_init_func = StructuredFunctionWrapper(
        init_func, "GeneratorDataset", input_classes=init_args_classes,
        input_shapes=init_args_shapes, input_types=init_args_types)
    self._state_classes = wrapped_init_func.output_classes
    self._state_shapes = wrapped_init_func.output_shapes
    self._state_types = wrapped_init_func.output_types
    self._init_func = wrapped_init_func.function

    wrapped_next_func = StructuredFunctionWrapper(
        next_func, "GeneratorDataset", input_classes=self._state_classes,
        input_shapes=self._state_shapes, input_types=self._state_types)
    self._output_classes = wrapped_next_func.output_classes
    self._output_shapes = wrapped_next_func.output_shapes
    self._output_types = wrapped_next_func.output_types
    self._next_func = wrapped_next_func.function

    wrapped_finalize_func = StructuredFunctionWrapper(
        finalize_func, "GeneratorDataset", input_classes=self._state_classes,
        input_shapes=self._state_shapes, input_types=self._state_types)
    self._finalize_func = wrapped_finalize_func.function

  def _as_variant_tensor(self):
    return gen_dataset_ops.generator_dataset(
        nest.flatten(self._init_args) + self._init_func.captured_inputs,
        self._next_func.captured_inputs,
        self._finalize_func.captured_inputs,
        init_func=self._init_func,
        next_func=self._next_func,
        finalize_func=self._finalize_func,
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


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
        **flat_structure(self))
    # pylint: enable=protected-access

  @property
  def output_classes(self):
    return nest.pack_sequence_as(
        self._datasets,
        [ds.output_classes for ds in nest.flatten(self._datasets)])

  @property
  def output_shapes(self):
    return nest.pack_sequence_as(
        self._datasets,
        [ds.output_shapes for ds in nest.flatten(self._datasets)])

  @property
  def output_types(self):
    return nest.pack_sequence_as(
        self._datasets,
        [ds.output_types for ds in nest.flatten(self._datasets)])


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
        **flat_structure(self))
    # pylint: enable=protected-access

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

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
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

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
    """Parse arguments according to the same rules as the `range()` builtin."""
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
        **flat_structure(self))

  @property
  def output_classes(self):
    return ops.Tensor

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
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class ShuffleDataset(Dataset):
  """A `Dataset` that randomly shuffles the elements of its input."""

  def __init__(self,
               input_dataset,
               buffer_size,
               seed=None,
               reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset.

    Args:
      input_dataset: The input dataset.
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

    Raises:
      ValueError: if invalid arguments are provided.
    """
    super(ShuffleDataset, self).__init__()
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._seed, self._seed2 = random_seed.get_seed(seed)
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
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

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
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

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
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class BatchDataset(Dataset):
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, drop_remainder):
    """See `Dataset.batch()` for details."""
    super(BatchDataset, self).__init__()
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

  def _as_variant_tensor(self):
    # TODO(jsimsa): Switch to using v2 only any time after 6/30/2018.
    if smart_cond.smart_constant_value(self._drop_remainder) is False:
      return gen_dataset_ops.batch_dataset(
          self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
          batch_size=self._batch_size,
          **flat_structure(self))
    else:
      return gen_dataset_ops.batch_dataset_v2(
          self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
          batch_size=self._batch_size,
          drop_remainder=self._drop_remainder,
          **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    input_shapes = self._input_dataset.output_shapes
    return nest.pack_sequence_as(input_shapes, [
        tensor_shape.vector(
            tensor_util.constant_value(self._batch_size) if smart_cond.
            smart_constant_value(self._drop_remainder) else None).concatenate(s)
        for s in nest.flatten(self._input_dataset.output_shapes)
    ])

  @property
  def output_types(self):
    return self._input_dataset.output_types


def _is_padded_shape_compatible_with(padded_shape, input_component_shape):
  """Returns `True` if `input_component_shape` can be padded to `padded_shape`.

  Args:
    padded_shape: A `tf.TensorShape`.
    input_component_shape: A `tf.TensorShape`.

  Returns:
    `True` if `input_component_shape` can be padded to `padded_shape`, otherwise
    `False`.
  """

  if padded_shape.dims is None or input_component_shape.dims is None:
    return True
  if len(padded_shape.dims) != len(input_component_shape.dims):
    return False
  for padded_dim, input_dim in zip(
      padded_shape.dims, input_component_shape.dims):
    if (padded_dim.value is not None and input_dim.value is not None
        and padded_dim.value < input_dim.value):
      return False
  return True


def _padded_shape_to_tensor(padded_shape, input_component_shape):
  """Converts `padded_shape` to a `tf.Tensor` representing that shape.

  Args:
    padded_shape: A shape-like object, which may be a `tf.TensorShape`, a Python
      sequence, or a 1-D `tf.Tensor` of `tf.int64` elements.
    input_component_shape: A `tf.TensorShape`, with which `padded_shape` must
      be compatible.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements, representing `padded_shape`.

  Raises:
    ValueError: If `padded_shape` is not a shape or not compatible with
      `input_component_shape`.
    TypeError: If `padded_shape` is not convertible to a `tf.int64` tensor.
  """
  try:
    # Try to convert the `padded_shape` to a `tf.TensorShape`
    padded_shape_as_shape = tensor_shape.as_shape(padded_shape)
    # We will return the "canonical" tensor representation, which uses
    # `-1` in place of `None`.
    ret = ops.convert_to_tensor(
        [dim if dim is not None else -1
         for dim in padded_shape_as_shape.as_list()], dtype=dtypes.int64)
  except (TypeError, ValueError):
    # The argument was not trivially convertible to a
    # `tf.TensorShape`, so fall back on the conversion to tensor
    # machinery.
    ret = ops.convert_to_tensor(padded_shape, preferred_dtype=dtypes.int64)
    if ret.shape.dims is not None and len(ret.shape.dims) != 1:
      raise ValueError(
          "Padded shape %s must be a 1-D tensor of tf.int64 values, but its "
          "shape was %s." % (padded_shape, ret.shape))
    if ret.dtype != dtypes.int64:
      raise TypeError(
          "Padded shape %s must be a 1-D tensor of tf.int64 values, but its "
          "element type was %s." % (padded_shape, ret.dtype.name))
    padded_shape_as_shape = tensor_util.constant_value_as_shape(ret)

  if not _is_padded_shape_compatible_with(padded_shape_as_shape,
                                          input_component_shape):
    raise ValueError("The padded shape %s is not compatible with the "
                     "corresponding input component shape %s."
                     % (padded_shape_as_shape, input_component_shape))

  return ret


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


def _default_padding(input_dataset):
  """Returns default padding tensors in a structure matching `input_dataset`."""
  def make_zero(t):
    if t.base_dtype == dtypes.string:
      return ""
    elif t.base_dtype == dtypes.variant:
      raise TypeError("Unable to create padding for field of type 'variant'")
    else:
      return np.zeros_like(t.as_numpy_dtype())

  return nest.map_structure(make_zero, input_dataset.output_types)


class PaddedBatchDataset(Dataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values,
               drop_remainder):
    """See `Dataset.batch()` for details."""
    super(PaddedBatchDataset, self).__init__()
    if sparse.any_sparse(input_dataset.output_classes):
      # TODO(b/63669786): support batching of sparse tensors
      raise TypeError(
          "Batching of padded sparse tensors is not currently supported")
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    padding_values = (
        padding_values
        if padding_values is not None else _default_padding(input_dataset))

    flat_padded_shapes = nest.flatten_up_to(input_dataset.output_shapes,
                                            padded_shapes)

    flat_padded_shapes_as_tensors = []

    for input_component_shape, padded_shape in zip(
        nest.flatten(input_dataset.output_shapes), flat_padded_shapes):
      flat_padded_shapes_as_tensors.append(
          _padded_shape_to_tensor(padded_shape, input_component_shape))

    self._padded_shapes = nest.pack_sequence_as(input_dataset.output_shapes,
                                                flat_padded_shapes_as_tensors)

    self._padding_values = nest.map_structure_up_to(
        input_dataset.output_shapes, _padding_value_to_tensor, padding_values,
        input_dataset.output_types)
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

  def _as_variant_tensor(self):
    # TODO(jsimsa): Switch to using v2 only any time after 6/30/2018.
    if smart_cond.smart_constant_value(self._drop_remainder) is False:
      return gen_dataset_ops.padded_batch_dataset(
          self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          output_shapes=nest.flatten(
              sparse.as_dense_shapes(self.output_shapes, self.output_classes)))
    else:
      return gen_dataset_ops.padded_batch_dataset_v2(
          self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          drop_remainder=self._drop_remainder,
          output_shapes=nest.flatten(
              sparse.as_dense_shapes(self.output_shapes, self.output_classes)))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):

    def _padded_shape_to_batch_shape(s):
      return tensor_shape.vector(
          tensor_util.constant_value(self._batch_size) if smart_cond.
          smart_constant_value(self._drop_remainder) else None).concatenate(
              tensor_util.constant_value_as_shape(s))

    return nest.map_structure(_padded_shape_to_batch_shape, self._padded_shapes)

  @property
  def output_types(self):
    return self._input_dataset.output_types


def _should_unpack_args(args):
  """Returns `True` if `args` should be `*args` when passed to a callable."""
  return type(args) is tuple  # pylint: disable=unidiomatic-typecheck


def _warn_if_collections(transformation_name):
  """Prints warning message if the current graph uses common graph collections.

  NOTE(mrry): Currently a warning is only generated for lookup tables. Any
  variables created will be automatically hoisted out to the outermost scope
  using `init_scope()`. Some collections (such as for control-flow contexts)
  are benign and should not generate a warning.

  Args:
    transformation_name: A human-readable name for the transformation.
  """
  if ops.get_default_graph().get_collection(ops.GraphKeys.TABLE_INITIALIZERS):
    warnings.warn("Creating lookup tables inside a function passed to %s is not"
                  " supported. Create each table outside the function, and "
                  "capture it inside the function to use it."
                  % transformation_name)


class MapDataset(Dataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.map()` for details."""
    super(MapDataset, self).__init__()
    self._input_dataset = input_dataset

    wrapped_func = StructuredFunctionWrapper(
        map_func, "Dataset.map()", input_dataset)
    self._output_classes = wrapped_func.output_classes
    self._output_shapes = wrapped_func.output_shapes
    self._output_types = wrapped_func.output_types
    self._map_func = wrapped_func.function

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    return gen_dataset_ops.map_dataset(
        input_t,
        self._map_func.captured_inputs,
        f=self._map_func,
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._output_classes

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
        **flat_structure(self))
    # pylint: enable=protected-access


class FlatMapDataset(Dataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    super(FlatMapDataset, self).__init__()
    self._input_dataset = input_dataset

    wrapped_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), input_dataset,
        experimental_nested_dataset_support=True)
    if not isinstance(wrapped_func.output_classes, _NestedDatasetComponent):
      raise TypeError("`map_func` must return a `Dataset` object.")
    self._output_classes = wrapped_func.output_classes.output_classes
    self._output_types = wrapped_func.output_types.output_types
    self._output_shapes = wrapped_func.output_shapes.output_shapes
    self._map_func = wrapped_func.function

  def _as_variant_tensor(self):
    return gen_dataset_ops.flat_map_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._map_func.captured_inputs,
        f=self._map_func,
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types

  def _transformation_name(self):
    return "Dataset.flat_map()"


class InterleaveDataset(FlatMapDataset):
  """A `Dataset` that maps a function over its input and interleaves the result.
  """

  def __init__(self, input_dataset, map_func, cycle_length, block_length):
    """See `Dataset.interleave()` for details."""
    super(InterleaveDataset, self).__init__(input_dataset, map_func)
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")

  def _as_variant_tensor(self):
    return gen_dataset_ops.interleave_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._map_func.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        f=self._map_func,  # pylint: disable=protected-access
        **flat_structure(self))

  def _transformation_name(self):
    return "Dataset.interleave()"


class FilterDataset(Dataset):
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self, input_dataset, predicate):
    """See `Dataset.filter()` for details."""
    super(FilterDataset, self).__init__()
    self._input_dataset = input_dataset
    wrapped_func = StructuredFunctionWrapper(
        predicate, "Dataset.filter()", input_dataset)
    if not (
        wrapped_func.output_types == dtypes.bool and
        wrapped_func.output_shapes.is_compatible_with(tensor_shape.scalar())):
      raise ValueError("`predicate` must return a scalar boolean tensor.")
    self._predicate = wrapped_func.function

  def _as_variant_tensor(self):
    return gen_dataset_ops.filter_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        other_arguments=self._predicate.captured_inputs,
        predicate=self._predicate,
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

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
    if buffer_size is None:
      buffer_size = -1  # This is the sentinel for auto-tuning.
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")

  def _as_variant_tensor(self):
    return gen_dataset_ops.prefetch_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        **flat_structure(self))

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types
