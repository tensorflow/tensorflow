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
import functools
import threading
import warnings

import numpy as np
import six

from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import filter_for_shard_ops
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import stats_options
from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import sparse
from tensorflow.python.data.util import structure as structure_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export


ops.NotDifferentiable("ReduceDataset")


@tf_export("data.Dataset", v1=[])
@six.add_metaclass(abc.ABCMeta)
class DatasetV2(object):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements (nested structures of tensors) and a "logical

  plan" of transformations that act on those elements.
  """

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

  @abc.abstractmethod
  def _inputs(self):
    """Returns a list of the input datasets of the dataset."""

    raise NotImplementedError("Dataset._inputs")

  def options(self):
    """Returns the options for this dataset and its inputs.

    Returns:
      A `tf.data.Options` object representing the dataset options.
    """
    options = Options()
    for input_dataset in self._inputs():
      input_options = input_dataset.options()
      if input_options is not None:
        options = options.merge(input_options)
    return options

  def _apply_options(self):
    """Apply options, such as optimization configuration, to the dataset."""

    dataset = self
    options = self.options()
    if options.experimental_threading is not None:
      t_options = options.experimental_threading
      if t_options.private_threadpool_size is not None:
        dataset = _PrivateThreadPoolDataset(dataset,
                                            t_options.private_threadpool_size)
      if t_options.max_intra_op_parallelism is not None:
        dataset = _MaxIntraOpParallelismDataset(
            dataset, t_options.max_intra_op_parallelism)
    static_optimizations = options._static_optimizations()  # pylint: disable=protected-access
    if static_optimizations:
      dataset = _OptimizeDataset(dataset, static_optimizations)
    if options.experimental_autotune is not False:
      dataset = _ModelDataset(dataset)
    if options.experimental_stats and options.experimental_stats.aggregator:  # pylint: disable=line-too-long
      dataset = _SetStatsAggregatorDataset(  # pylint: disable=protected-access
          dataset, options.experimental_stats.aggregator,
          options.experimental_stats.prefix,
          options.experimental_stats.counter_prefix)
    return dataset

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
    dataset = self._apply_options()
    if shared_name is None:
      shared_name = ""
    if compat.forward_compatible(2018, 8, 3):
      iterator_resource = gen_dataset_ops.iterator_v2(
          container="", shared_name=shared_name, **flat_structure(self))
    else:
      iterator_resource = gen_dataset_ops.iterator(
          container="", shared_name=shared_name, **flat_structure(self))
    with ops.colocate_with(iterator_resource):
      initializer = gen_dataset_ops.make_iterator(
          dataset._as_variant_tensor(),  # pylint: disable=protected-access
          iterator_resource)
    return iterator_ops.Iterator(iterator_resource, initializer,
                                 dataset.output_types, dataset.output_shapes,
                                 dataset.output_classes)

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
      dataset = self._apply_options()
      return iterator_ops.EagerIterator(dataset)
    else:
      raise RuntimeError("dataset.__iter__() is only supported when eager "
                         "execution is enabled.")

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
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

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
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

    Args:
      tensors: A nested structure of tensors, each having the same size in the
        0th dimension.

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorSliceDataset(tensors)

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
    tf.enable_eager_execution()

    def gen():
      for i in itertools.count(1):
        yield (i, [1] * i)

    ds = tf.data.Dataset.from_generator(
        gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

    for value in ds.take(2):
      print value
    # (1, array([1]))
    # (2, array([1, 1]))
    ```

    NOTE: The current implementation of `Dataset.from_generator()` uses
    `tf.py_func` and inherits the same constraints. In particular, it
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

    generator_state = DatasetV2._GeneratorState(generator)

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
      *args: follows the same semantics as python's xrange.
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
    """A dataset of all files matching one or more glob patterns.

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
      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.set_random_seed` for behavior.

    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    """
    with ops.name_scope("list_files"):
      if shuffle is None:
        shuffle = True
      file_pattern = ops.convert_to_tensor(
          file_pattern, dtype=dtypes.string, name="file_pattern")
      matching_files = gen_io_ops.matching_files(file_pattern)

      # Raise an exception if `file_pattern` does not match any files.
      condition = math_ops.greater(array_ops.shape(matching_files)[0], 0,
                                   name="match_not_empty")

      message = math_ops.add(
          "No files matched pattern: ",
          string_ops.reduce_join(file_pattern, separator=", "), name="message")

      assert_not_empty = control_flow_ops.Assert(
          condition, [message], summarize=1, name="assert_not_empty")
      with ops.control_dependencies([assert_not_empty]):
        matching_files = array_ops.identity(matching_files)

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

    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new
        dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        random seed that will be used to create the distribution. See
        `tf.set_random_seed` for behavior.
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
        whether the last batch should be dropped in the case it has fewer than
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

    Like `tf.data.Dataset.batch`, the tensors in the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padding_shapes`. The `padding_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant (e.g. `tf.Dimension(37)`), the component
      will be padded out to that length in that dimension.
    * If the dimension is unknown (e.g. `tf.Dimension(None)`), the component
      will be padded out to the maximum length of all elements in that
      dimension.

    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.SparseTensor`.

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
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    """
    return PaddedBatchDataset(self, batch_size, padded_shapes, padding_values,
                              drop_remainder)

  def map(self, map_func, num_parallel_calls=None):
    """Maps `map_func` across the elements of this dataset.

    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input.

    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3, 4, 5 }

    a.map(lambda x: x + 1) = { 2, 3, 4, 5, 6 }
    ```

    The input signature of `map_func` is determined by the structure of each
    element in this dataset. For example:

    ```python
    # Each element is a `tf.Tensor` object.
    a = { 1, 2, 3, 4, 5 }
    # `map_func` takes a single argument of type `tf.Tensor` with the same
    # shape and dtype.
    result = a.map(lambda x: ...)

    # Each element is a tuple containing two `tf.Tensor` objects.
    b = { (1, "foo"), (2, "bar"), (3, "baz") }
    # `map_func` takes two arguments of type `tf.Tensor`.
    result = b.map(lambda x_int, y_str: ...)

    # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    c = { {"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"} }
    # `map_func` takes a single argument of type `dict` with the same keys as
    # the elements.
    result = c.map(lambda d: ...)
    ```

    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.

    ```python
    # `map_func` returns a scalar `tf.Tensor` of type `tf.float32`.
    def f(...):
      return tf.constant(37.0)
    result = dataset.map(f)
    result.output_classes == tf.Tensor
    result.output_types == tf.float32
    result.output_shapes == []  # scalar

    # `map_func` returns two `tf.Tensor` objects.
    def g(...):
      return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
    result = dataset.map(g)
    result.output_classes == (tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string)
    result.output_shapes == ([], [3])

    # Python primitives, lists, and NumPy arrays are implicitly converted to
    # `tf.Tensor`.
    def h(...):
      return 37.0, ["Foo", "Bar", "Baz"], np.array([1.0, 2.0] dtype=np.float64)
    result = dataset.map(h)
    result.output_classes == (tf.Tensor, tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string, tf.float64)
    result.output_shapes == ([], [3], [2])

    # `map_func` can return nested structures.
    def i(...):
      return {"a": 37.0, "b": [42, 16]}, "foo"
    result.output_classes == ({"a": tf.Tensor, "b": tf.Tensor}, tf.Tensor)
    result.output_types == ({"a": tf.float32, "b": tf.int32}, tf.string)
    result.output_shapes == ({"a": [], "b": [2]}, [])
    ```

    In addition to `tf.Tensor` objects, `map_func` can accept as arguments and
    return `tf.SparseTensor` objects.

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

    Use `flat_map` if you want to make sure that the order of your dataset
    stays the same. For example, to flatten a dataset of batches into a
    dataset of their elements:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset. '[...]' represents a tensor.
    a = {[1,2,3,4,5], [6,7,8,9], [10]}

    a.flat_map(lambda x: Dataset.from_tensor_slices(x)) ==
      {[1,2,3,4,5,6,7,8,9,10]}
    ```

    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
    `flat_map` produces the same output as
    `tf.data.Dataset.interleave(cycle_length=1)`

    Args:
      map_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        `Dataset`.

    Returns:
      Dataset: A `Dataset`.
    """
    return FlatMapDataset(self, map_func)

  def interleave(self,
                 map_func,
                 cycle_length,
                 block_length=1,
                 num_parallel_calls=None):
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
    identical results to `tf.data.Dataset.flat_map`. In general,
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
      num_parallel_calls: (Optional.) If specified, the implementation creates
        a threadpool, which is used to fetch inputs from cycle elements
        asynchronously and in parallel. The default behavior is to fetch inputs
        from cycle elements synchronously with no parallelism.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None:
      return InterleaveDataset(self, map_func, cycle_length, block_length)
    else:
      return ParallelInterleaveDataset(self, map_func, cycle_length,
                                       block_length, num_parallel_calls)

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
    """Applies a transformation function to this dataset.

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
    if not isinstance(dataset, DatasetV2):
      raise TypeError("`transformation_func` must return a Dataset.")
    dataset._input_datasets = [self]  # pylint: disable=protected-access
    return dataset

  def window(self, size, shift=None, stride=1, drop_remainder=False):
    """Combines input elements into a dataset of windows.

    Each window is a dataset itself and contains `size` elements (or
    possibly fewer if there are not enough input elements to fill the window
    and `drop_remainder` evaluates to false).

    The `stride` argument determines the stride of the input elements,
    and the `shift` argument determines the shift of the window.

    For example:
    - `tf.data.Dataset.range(7).window(2)` produces
      `{{0, 1}, {2, 3}, {4, 5}, {6}}`
    - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
      `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
    - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
      `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

    Args:
      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
        of the input dataset to combine into a window.
      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        forward shift of the sliding window in each iteration. Defaults to
        `size`.
      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        stride of the input elements in the sliding window.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether a window should be dropped in case its size is smaller than
        `window_size`.

    Returns:
      Dataset: A `Dataset` of windows, each of which is a nested `Dataset` with
        the same structure as this dataset, but a finite subsequence of its
        elements.
    """
    if shift is None:
      shift = size
    return WindowDataset(self, size, shift, stride, drop_remainder)

  def reduce(self, initial_state, reduce_func):
    """Reduces the input dataset to a single element.

    The transformation calls `reduce_func` successively on every element of
    the input dataset until the dataset is exhausted, aggregating information in
    its internal state. The `initial_state` argument is used for the initial
    state and the final state is returned as the result.

    For example:
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1)`
      produces `5`
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y)`
      produces `10`

    Args:
      initial_state: A nested structure of tensors, representing the initial
        state of the transformation.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. It must take two arguments and return a nested structure
        of tensors. The structure of `new_state` must match the structure of
        `initial_state`.

    Returns:
      A nested structure of `tf.Tensor` objects, corresponding to the final
      state of the transformation.

    """

    with ops.name_scope("initial_state"):
      # Convert any `SparseTensorValue`s to `SparseTensor`s and all other
      # values to tensors.
      initial_state = nest.pack_sequence_as(initial_state, [
          sparse_tensor_lib.SparseTensor.from_value(t)
          if sparse_tensor_lib.is_sparse(t) else ops.convert_to_tensor(
              t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(initial_state))
      ])

    # Compute initial values for the state classes, shapes and types based on
    # the initial state.
    state_classes = sparse.get_classes(initial_state)
    state_shapes = nest.pack_sequence_as(
        initial_state, [t.get_shape() for t in nest.flatten(initial_state)])
    state_types = nest.pack_sequence_as(
        initial_state, [t.dtype for t in nest.flatten(initial_state)])

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `self._state_shapes`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = StructuredFunctionWrapper(
          reduce_func,
          "reduce()",
          input_classes=(state_classes, self.output_classes),
          input_shapes=(state_shapes, self.output_shapes),
          input_types=(state_types, self.output_types),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      output_classes = wrapped_func.output_classes
      for new_state_class, state_class in zip(
          nest.flatten(output_classes), nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." % (state_classes,
                                               wrapped_func.output_classes))

      # Extract and validate type information from the returned values.
      output_types = wrapped_func.output_types
      for new_state_type, state_type in zip(
          nest.flatten(output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." % (state_types,
                                               wrapped_func.output_types))

      # Extract shape information from the returned values.
      output_shapes = wrapped_func.output_shapes
      flat_state_shapes = nest.flatten(state_shapes)
      flat_new_state_shapes = nest.flatten(output_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        state_shapes = nest.pack_sequence_as(state_shapes,
                                             weakened_state_shapes)

    reduce_func = wrapped_func.function
    reduce_func.add_to_graph(ops.get_default_graph())

    return sparse.deserialize_sparse_tensors(
        nest.pack_sequence_as(
            output_types,
            gen_dataset_ops.reduce_dataset(
                self._as_variant_tensor(),  # pylint: disable=protected-access
                nest.flatten(sparse.serialize_sparse_tensors(initial_state)),
                reduce_func.captured_inputs,
                f=reduce_func,
                output_shapes=nest.flatten(
                    sparse.as_dense_shapes(output_shapes, output_classes)),
                output_types=nest.flatten(
                    sparse.as_dense_types(output_types, output_classes)))),
        output_types,
        output_shapes,
        output_classes)

  def with_options(self, options):
    """Returns a new `tf.data.Dataset` with the given options set.

    The options are "global" in the sense they apply to the entire dataset.
    If options are set multiple times, they are merged as long as different
    options do not use different non-default values.

    Args:
      options: A `tf.data.Options` that identifies the options the use.

    Returns:
      Dataset: A `Dataset` with the given options.

    Raises:
      ValueError: when an option is set more than once to a non-default value
    """
    return _OptionsDataset(self, options)


@tf_export(v1=["data.Dataset"])
class DatasetV1(DatasetV2):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements (nested structures of tensors) and a "logical
  plan" of transformations that act on those elements.
  """

  def __init__(self):
    pass

  @deprecation.deprecated(
      None, "Use `for ... in dataset:` to iterate over a dataset. If using "
      "`tf.estimator`, return the `Dataset` object directly from your input "
      "function. As a last resort, you can use "
      "`tf.compat.v1.data.make_one_shot_iterator(dataset)`.")
  def make_one_shot_iterator(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.
    """
    if context.executing_eagerly():
      dataset = self._apply_options()
      return iterator_ops.EagerIterator(dataset)

    graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(capture_by_value=True)
    def _make_dataset():
      """Factory function for a dataset."""
      # NOTE(mrry): `Defun` does not capture the graph-level seed from the
      # enclosing graph, so if a graph-level seed is present we set the local
      # graph seed based on a combination of the graph- and op-level seeds.
      if graph_level_seed is not None:
        assert op_level_seed is not None
        core_random_seed.set_random_seed(
            (graph_level_seed + 87654321 * op_level_seed) % (2 ** 63 - 1))

      dataset = self._apply_options()
      return dataset._as_variant_tensor()  # pylint: disable=protected-access

    try:
      _make_dataset.add_to_graph(ops.get_default_graph())
    except ValueError as err:
      if "Cannot capture a stateful node" in str(err):
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

  @staticmethod
  @functools.wraps(DatasetV2.from_tensors)
  def from_tensors(tensors):
    return DatasetV1Adapter(DatasetV2.from_tensors(tensors))

  @staticmethod
  @functools.wraps(DatasetV2.from_tensor_slices)
  def from_tensor_slices(tensors):
    return DatasetV1Adapter(DatasetV2.from_tensor_slices(tensors))

  @staticmethod
  @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
  def from_sparse_tensor_slices(sparse_tensor):
    """Splits each rank-N `tf.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
    return DatasetV1Adapter(SparseTensorSliceDataset(sparse_tensor))

  @staticmethod
  @functools.wraps(DatasetV2.from_generator)
  def from_generator(generator, output_types, output_shapes=None, args=None):
    return DatasetV1Adapter(DatasetV2.from_generator(
        generator, output_types, output_shapes, args))

  @staticmethod
  @functools.wraps(DatasetV2.range)
  def range(*args):
    return DatasetV1Adapter(DatasetV2.range(*args))

  @staticmethod
  @functools.wraps(DatasetV2.zip)
  def zip(datasets):
    return DatasetV1Adapter(DatasetV2.zip(datasets))

  @functools.wraps(DatasetV2.concatenate)
  def concatenate(self, dataset):
    return DatasetV1Adapter(super(DatasetV1, self).concatenate(dataset))

  @functools.wraps(DatasetV2.prefetch)
  def prefetch(self, buffer_size):
    return DatasetV1Adapter(super(DatasetV1, self).prefetch(buffer_size))

  @staticmethod
  @functools.wraps(DatasetV2.list_files)
  def list_files(file_pattern, shuffle=None, seed=None):
    return DatasetV1Adapter(DatasetV2.list_files(file_pattern, shuffle, seed))

  @functools.wraps(DatasetV2.repeat)
  def repeat(self, count=None):
    return DatasetV1Adapter(super(DatasetV1, self).repeat(count))

  @functools.wraps(DatasetV2.shuffle)
  def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
    return DatasetV1Adapter(super(DatasetV1, self).shuffle(
        buffer_size, seed, reshuffle_each_iteration))

  @functools.wraps(DatasetV2.cache)
  def cache(self, filename=""):
    return DatasetV1Adapter(super(DatasetV1, self).cache(filename))

  @functools.wraps(DatasetV2.take)
  def take(self, count):
    return DatasetV1Adapter(super(DatasetV1, self).take(count))

  @functools.wraps(DatasetV2.skip)
  def skip(self, count):
    return DatasetV1Adapter(super(DatasetV1, self).skip(count))

  @deprecation.deprecated(
      None, "Use `dataset.apply(tf.data.experimental.filter_for_shard(...))`.")
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
        checking is done on a best-effort basis, and errors aren't guaranteed
        to be caught upon dataset creation. (e.g. providing in a placeholder
        tensor bypasses the early checking, and will instead result in an error
        during a session.run call.)
    """
    return self.apply(filter_for_shard_ops.filter_for_shard(num_shards, index))

  @functools.wraps(DatasetV2.batch)
  def batch(self, batch_size, drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).batch(
        batch_size, drop_remainder))

  @functools.wraps(DatasetV2.padded_batch)
  def padded_batch(self,
                   batch_size,
                   padded_shapes,
                   padding_values=None,
                   drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).padded_batch(
        batch_size, padded_shapes, padding_values, drop_remainder))

  @functools.wraps(DatasetV2.map)
  def map(self, map_func, num_parallel_calls=None):
    return DatasetV1Adapter(super(DatasetV1, self).map(
        map_func, num_parallel_calls))

  @functools.wraps(DatasetV2.flat_map)
  def flat_map(self, map_func):
    return DatasetV1Adapter(super(DatasetV1, self).flat_map(map_func))

  @functools.wraps(DatasetV2.interleave)
  def interleave(self,
                 map_func,
                 cycle_length,
                 block_length=1,
                 num_parallel_calls=None):
    return DatasetV1Adapter(super(DatasetV1, self).interleave(
        map_func, cycle_length, block_length, num_parallel_calls))

  @functools.wraps(DatasetV2.filter)
  def filter(self, predicate):
    return DatasetV1Adapter(super(DatasetV1, self).filter(predicate))

  @functools.wraps(DatasetV2.apply)
  def apply(self, transformation_func):
    return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

  @functools.wraps(DatasetV2.window)
  def window(self, size, shift=None, stride=1, drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).window(
        size, shift, stride, drop_remainder))

  @functools.wraps(DatasetV2.with_options)
  def with_options(self, options):
    return DatasetV1Adapter(super(DatasetV1, self).with_options(options))


# TODO(b/119044825): Until all `tf.data` unit tests are converted to V2, keep
# this alias in place.
Dataset = DatasetV1


class DatasetV1Adapter(DatasetV1):
  """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

  def __init__(self, dataset):
    super(DatasetV1Adapter, self).__init__()
    self._dataset = dataset

  def _as_variant_tensor(self):
    return self._dataset._as_variant_tensor()  # pylint: disable=protected-access

  def _inputs(self):
    return self._dataset._inputs()  # pylint: disable=protected-access

  def options(self):
    return self._dataset.options()

  @property
  def output_classes(self):
    return self._dataset.output_classes

  @property
  def output_shapes(self):
    return self._dataset.output_shapes

  @property
  def output_types(self):
    return self._dataset.output_types

  def make_initializable_iterator(self, shared_name=None):
    return self._dataset.make_initializable_iterator(shared_name)

  def __iter__(self):
    return iter(self._dataset)


@tf_export(v1=["data.make_one_shot_iterator"])
def make_one_shot_iterator(dataset):
  """Creates a `tf.data.Iterator` for enumerating the elements of a dataset.

  Note: The returned iterator will be initialized automatically.
  A "one-shot" iterator does not support re-initialization.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A `tf.data.Iterator` over the elements of this dataset.
  """
  try:
    # Call the defined `make_one_shot_iterator()` if there is one, because some
    # datasets (e.g. for prefetching) override its behavior.
    return dataset.make_one_shot_iterator()
  except AttributeError:
    return DatasetV1Adapter(dataset).make_one_shot_iterator()


@tf_export("data.Options")
class Options(options_lib.OptionsBase):
  """Represents options for tf.data.Dataset.

  An `Options` object can be, for instance, used to control which static
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations such as `tf.data.Dataset.map` or
  `tf.data.Dataset.interleave`.
  """

  experimental_autotune = options_lib.create_option(
      name="experimental_autotune",
      ty=bool,
      docstring=
      "Whether to dynamically adjust the values of tunable parameters (e.g. "
      "degrees of parallelism).")

  experimental_deterministic = options_lib.create_option(
      name="experimental_deterministic",
      ty=bool,
      docstring=
      "Whether to dynamically adjust the values of tunable parameters (e.g. "
      "degrees of parallelism).")

  experimental_numa_aware = options_lib.create_option(
      name="experimental_numa_aware",
      ty=bool,
      docstring="Whether to use NUMA-aware operations.")

  experimental_optimization = options_lib.create_option(
      name="experimental_optimization",
      ty=optimization_options.OptimizationOptions,
      docstring="Associates the given optimization options with the dataset.")

  experimental_stats = options_lib.create_option(
      name="experimental_stats",
      ty=stats_options.StatsOptions,
      docstring="Associates the given statistics options with the dataset.")

  experimental_threading = options_lib.create_option(
      name="experimental_threading",
      ty=threading_options.ThreadingOptions,
      docstring="Associates the given threading options with the dataset.")

  def _static_optimizations(self):
    """Produces the list of enabled static optimizations."""

    result = []
    exp_optimization_options = self.experimental_optimization
    if exp_optimization_options:
      optimizations = [
          "filter_fusion",
          "hoist_random_uniform",
          "map_and_batch_fusion",
          "map_and_filter_fusion",
          "map_fusion",
          "map_parallelization",
          "map_vectorization",
          "noop_elimination",
          "shuffle_and_repeat_fusion",
      ]
      for optimization in optimizations:
        if getattr(exp_optimization_options, optimization):
          result.append(optimization)
    if self.experimental_numa_aware:
      result.append("make_numa_aware")
    if self.experimental_deterministic is False:
      result.append("make_sloppy")
    exp_stats_options = self.experimental_stats
    if exp_stats_options and exp_stats_options.latency_all_edges:
      result.append("latency_all_edges")
    return result

  def merge(self, options):
    """Merges itself with the given `tf.data.Options`.

    The given `tf.data.Options` can be merged as long as there does not exist an
    attribute that is set to different values in `self` and `options`.

    Args:
      options: a `tf.data.Options` to merge with

    Raises:
      ValueError: if the given `tf.data.Options` cannot be merged

    Returns:
      New `tf.data.Options()` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)


class DatasetSource(DatasetV2):
  """Abstract class representing a dataset with no inputs."""

  def _inputs(self):
    return []


class UnaryDataset(DatasetV2):
  """Abstract class representing a dataset with one input."""

  def __init__(self, input_dataset):
    super(UnaryDataset, self).__init__()
    self._input_dataset = input_dataset

  def _inputs(self):
    return [self._input_dataset]


class UnaryUnchangedStructureDataset(UnaryDataset):
  """Represents a unary dataset with the same input and output structure."""

  @property
  def output_classes(self):
    return self._input_dataset.output_classes  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._input_dataset.output_types  # pylint: disable=protected-access


class TensorDataset(DatasetSource):
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


class TensorSliceDataset(DatasetSource):
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

    batch_dim = tensor_shape.Dimension(tensor_shape.dimension_value(
        flat_tensors[0].get_shape()[0]))
    for t in flat_tensors[1:]:
      batch_dim.assert_is_compatible_with(tensor_shape.Dimension(
          tensor_shape.dimension_value(t.get_shape()[0])))
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


class SparseTensorSliceDataset(DatasetSource):
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
    rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
    num_values = tensor_shape.Dimension(None)
    return (tensor_shape.TensorShape([num_values, rank]),
            tensor_shape.TensorShape([num_values]),
            tensor_shape.TensorShape([rank]))

  @property
  def output_types(self):
    return (dtypes.int64, self._sparse_tensor.dtype, dtypes.int64)


class _VariantDataset(DatasetV2):
  """A Dataset wrapper around a `tf.variant`-typed function argument."""

  def __init__(self, dataset_variant, structure):
    super(_VariantDataset, self).__init__()
    self._dataset_variant = dataset_variant
    self._structure = structure

  def _as_variant_tensor(self):
    return self._dataset_variant

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return self._structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._structure._to_legacy_output_types()  # pylint: disable=protected-access


class DatasetStructure(structure_lib.Structure):
  """Represents a `Dataset` of structured values."""

  def __init__(self, element_structure):
    self._element_structure = element_structure

  @property
  def _flat_shapes(self):
    return [tensor_shape.scalar()]

  @property
  def _flat_types(self):
    return [dtypes.variant]

  def is_compatible_with(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, DatasetStructure) and
            self._element_structure.is_compatible_with(
                other._element_structure))

  def _to_tensor_list(self, value):
    return [value._as_variant_tensor()]  # pylint: disable=protected-access

  def _from_tensor_list(self, flat_value):
    if (len(flat_value) != 1 or flat_value[0].dtype != dtypes.variant or
        not flat_value[0].shape.is_compatible_with(tensor_shape.scalar())):
      raise ValueError(
          "DatasetStructure corresponds to a single tf.variant scalar.")
    return self._from_compatible_tensor_list(flat_value)

  def _from_compatible_tensor_list(self, flat_value):
    # pylint: disable=protected-access
    return _VariantDataset(flat_value[0], self._element_structure)

  @staticmethod
  def from_value(value):
    # TODO(b/110122868): We can simplify this when a `Dataset` object has a
    # `Structure`-valued property.
    element_structure = structure_lib.Structure._from_legacy_structure(
        value.output_types, value.output_shapes, value.output_classes)
    return DatasetStructure(element_structure)

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self

  def _batch(self, batch_size):
    raise NotImplementedError("Batching for `tf.data.Dataset` objects.")


# pylint: disable=protected-access
structure_lib.Structure._register_custom_converter(DatasetV2,
                                                   DatasetStructure.from_value)
# pylint: enable=protected-access


class StructuredFunctionWrapper(object):
  """A wrapper for `Defun` that supports structured arguments and return values.
  """

  def __init__(self,
               func,
               transformation_name,
               dataset=None,
               input_classes=None,
               input_shapes=None,
               input_types=None,
               add_to_graph=True,
               defun_kwargs=None):
    """Creates a new `StructuredFunctionWrapper` for the given function.

    Args:
      func: A function from a nested structure to another nested structure.
      transformation_name: Human-readable name of the transformation in which
        this function is being instantiated, for error messages.
      dataset: (Optional.) A `tf.data.Dataset`. If given, the structure of this
        dataset will be assumed as the structure for `func` arguments; otherwise
        `input_classes`, `input_shapes`, and `input_types` must be defined.
      input_classes: (Optional.) A nested structure of `type`. If given, this
        argument defines the Python types for `func` arguments.
      input_shapes: (Optional.) A nested structure of `tf.TensorShape`. If
        given, this argument defines the shapes and structure for `func`
        arguments.
      input_types: (Optional.) A nested structure of `tf.DType`. If given, this
        argument defines the element types and structure for `func` arguments.
      add_to_graph: (Optional.) If `True`, the function will be added to the
        default graph.
      defun_kwargs: (Optional.) A dictionary mapping string argument names to
        values. If supplied, will be passed to `function.Defun()` as keyword
        arguments.

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

    self._input_structure = structure_lib.Structure._from_legacy_structure(  # pylint: disable=protected-access
        self._input_types, self._input_shapes, self._input_classes)

    self._transformation_name = transformation_name
    readable_transformation_name = transformation_name.replace(
        ".", "_")[:-2] if len(transformation_name) > 2 else ""
    self._func_name = "_".join([
        readable_transformation_name,
        function_utils.get_func_name(func),
        str(ops.uid())
    ])

    if defun_kwargs is None:
      defun_kwargs = {}

    @function.Defun(
        *self._input_structure._flat_types, func_name=self._func_name,  # pylint: disable=protected-access
        **defun_kwargs)
    def tf_data_structured_function_wrapper(*args):
      """Wrapper for passing nested structures to and from tf.data functions."""
      # pylint: disable=protected-access
      nested_args = self._input_structure._from_compatible_tensor_list(args)
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

      try:
        self._output_structure = structure_lib.Structure.from_value(ret)
      except (ValueError, TypeError):
        raise TypeError("Unsupported return value from function passed to "
                        "%s: %s." % (transformation_name, ret))

      _warn_if_collections(transformation_name)
      return self._output_structure._to_tensor_list(ret)

    self._function = tf_data_structured_function_wrapper
    if add_to_graph:
      self._function.add_to_graph(ops.get_default_graph())
    else:
      # Use the private method that will execute
      # `tf_data_structured_function_wrapper` but delay adding it to the graph
      # in case (e.g.) we need to rerun the function.
      self._function._create_definition_if_needed()  # pylint: disable=protected-access

  @property
  def output_structure(self):
    return self._output_structure

  @property
  def output_classes(self):
    return self._output_structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._output_structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._output_structure._to_legacy_output_types()  # pylint: disable=protected-access

  @property
  def function(self):
    return self._function


def flat_structure(dataset=None, structure=None):
  """Helper for setting `output_shapes` and `output_types` attrs of Dataset ops.

  Either `dataset` or `structure` must be passed to this function.

  Most Dataset op constructors expect `output_shapes` and `output_types`
  arguments that represent the flattened structure of an element. This helper
  function generates these attrs as a keyword argument dictionary, allowing
  `Dataset._as_variant_tensor()` implementations to pass
  `**flat_structure(self)` to the op constructor.

  Args:
    dataset: (Optional.) A `tf.data.Dataset`.
    structure: (Optional.) A `Structure`.

  Returns:
    A dictionary of keyword arguments that can be passed to many Dataset op
    constructors.
  """
  # pylint: disable=protected-access
  if structure is None:
    structure = structure_lib.Structure._from_legacy_structure(
        dataset.output_types, dataset.output_shapes, dataset.output_classes)
  return {
      "output_shapes": structure._flat_shapes,
      "output_types": structure._flat_types,
  }


class _GeneratorDataset(DatasetSource):
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
        init_func,
        self._transformation_name(),
        input_classes=init_args_classes,
        input_shapes=init_args_shapes,
        input_types=init_args_types)
    self._state_classes = wrapped_init_func.output_classes
    self._state_shapes = wrapped_init_func.output_shapes
    self._state_types = wrapped_init_func.output_types
    self._init_func = wrapped_init_func.function

    wrapped_next_func = StructuredFunctionWrapper(
        next_func,
        self._transformation_name(),
        input_classes=self._state_classes,
        input_shapes=self._state_shapes,
        input_types=self._state_types)
    self._output_classes = wrapped_next_func.output_classes
    self._output_shapes = wrapped_next_func.output_shapes
    self._output_types = wrapped_next_func.output_types
    self._next_func = wrapped_next_func.function

    wrapped_finalize_func = StructuredFunctionWrapper(
        finalize_func,
        self._transformation_name(),
        input_classes=self._state_classes,
        input_shapes=self._state_shapes,
        input_types=self._state_types)
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

  def _transformation_name(self):
    return "Dataset.from_generator()"


class ZipDataset(DatasetV2):
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets):
    """See `Dataset.zip()` for details."""
    super(ZipDataset, self).__init__()
    for ds in nest.flatten(datasets):
      if not isinstance(ds, DatasetV2):
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

  def _inputs(self):
    return nest.flatten(self._datasets)

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


class ConcatenateDataset(DatasetV2):
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate):
    """See `Dataset.concatenate()` for details."""
    super(ConcatenateDataset, self).__init__()
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate

    self._output_types = input_dataset.output_types
    if self._output_types != dataset_to_concatenate.output_types:
      raise TypeError(
          "Two datasets to concatenate have different types %s and %s" %
          (self._output_types, dataset_to_concatenate.output_types))

    self._output_classes = input_dataset.output_classes
    if self._output_classes != dataset_to_concatenate.output_classes:
      raise TypeError(
          "Two datasets to concatenate have different classes %s and %s" %
          (self._output_classes, dataset_to_concatenate.output_classes))

    input_shapes = self._input_dataset.output_shapes
    self._output_shapes = nest.pack_sequence_as(input_shapes, [
        ts1.most_specific_compatible_shape(ts2)
        for (ts1, ts2) in zip(
            nest.flatten(input_shapes),
            nest.flatten(self._dataset_to_concatenate.output_shapes))
    ])

    self._input_datasets = [input_dataset, dataset_to_concatenate]

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.concatenate_dataset(
        self._input_dataset._as_variant_tensor(),
        self._dataset_to_concatenate._as_variant_tensor(),
        **flat_structure(self))
    # pylint: enable=protected-access

  def _inputs(self):
    return [self._input_dataset, self._dataset_to_concatenate]

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class RepeatDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count):
    """See `Dataset.repeat()` for details."""
    super(RepeatDataset, self).__init__(input_dataset)
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


class RangeDataset(DatasetSource):
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


class CacheDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename):
    """See `Dataset.cache()` for details."""
    super(CacheDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")

  def _as_variant_tensor(self):
    return gen_dataset_ops.cache_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        filename=self._filename,
        **flat_structure(self))


class ShuffleDataset(UnaryUnchangedStructureDataset):
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
        `tf.set_random_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if invalid arguments are provided.
    """
    super(ShuffleDataset, self).__init__(input_dataset)
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


class TakeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` containing the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.take()` for details."""
    super(TakeDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")

  def _as_variant_tensor(self):
    return gen_dataset_ops.take_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        count=self._count,
        **flat_structure(self))


class SkipDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.skip()` for details."""
    super(SkipDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")

  def _as_variant_tensor(self):
    return gen_dataset_ops.skip_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        count=self._count,
        **flat_structure(self))


class BatchDataset(UnaryDataset):
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, drop_remainder):
    """See `Dataset.batch()` for details."""
    super(BatchDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    # pylint: disable=protected-access
    input_structure = structure_lib.Structure._from_legacy_structure(
        input_dataset.output_types, input_dataset.output_shapes,
        input_dataset.output_classes)
    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      self._output_structure = input_structure._batch(
          tensor_util.constant_value(self._batch_size))
    else:
      self._output_structure = input_structure._batch(None)

  def _as_variant_tensor(self):
    return gen_dataset_ops.batch_dataset_v2(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        batch_size=self._batch_size,
        drop_remainder=self._drop_remainder,
        **flat_structure(structure=self._output_structure))

  @property
  def output_classes(self):
    return self._output_structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._output_structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._output_structure._to_legacy_output_types()  # pylint: disable=protected-access


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


class PaddedBatchDataset(UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values,
               drop_remainder):
    """See `Dataset.batch()` for details."""
    super(PaddedBatchDataset, self).__init__(input_dataset)
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


class MapDataset(UnaryDataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self, input_dataset, map_func, use_inter_op_parallelism=True):
    """See `Dataset.map()` for details."""
    super(MapDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    return gen_dataset_ops.map_dataset(
        input_t,
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        **flat_structure(structure=self._map_func.output_structure))

  @property
  def output_classes(self):
    return self._map_func.output_classes

  @property
  def output_shapes(self):
    return self._map_func.output_shapes

  @property
  def output_types(self):
    return self._map_func.output_types

  def _transformation_name(self):
    return "Dataset.map()"


class ParallelMapDataset(MapDataset):
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               use_inter_op_parallelism=True):
    """See `Dataset.map()` for details."""
    super(ParallelMapDataset, self).__init__(input_dataset, map_func,
                                             use_inter_op_parallelism)

    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int32, name="num_parallel_calls")

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    input_t = self._input_dataset._as_variant_tensor()
    return gen_dataset_ops.parallel_map_dataset(
        input_t,
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        num_parallel_calls=self._num_parallel_calls,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        **flat_structure(structure=self._map_func.output_structure))


class FlatMapDataset(UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    super(FlatMapDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset

    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetStructure):
      raise TypeError("`map_func` must return a `Dataset` object.")
    self._output_structure = self._map_func.output_structure._element_structure  # pylint: disable=protected-access

  def _as_variant_tensor(self):
    return gen_dataset_ops.flat_map_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **flat_structure(structure=self._output_structure))

  @property
  def output_classes(self):
    return self._output_structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    return self._output_structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._output_structure._to_legacy_output_types()  # pylint: disable=protected-access

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
    # pylint: disable=protected-access
    return gen_dataset_ops.interleave_dataset(
        self._input_dataset._as_variant_tensor(),
        self._map_func.function.captured_inputs,
        self._cycle_length,
        self._block_length,
        f=self._map_func.function,
        **flat_structure(structure=self._output_structure))

  def _transformation_name(self):
    return "Dataset.interleave()"


class ParallelInterleaveDataset(FlatMapDataset):
  """A `Dataset` that maps a function over its input and interleaves the result.

  """

  def __init__(self, input_dataset, map_func, cycle_length, block_length,
               num_parallel_calls):
    """See `Dataset.interleave()` for details."""
    super(ParallelInterleaveDataset, self).__init__(input_dataset, map_func)
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    return gen_dataset_ops.parallel_interleave_dataset_v2(
        self._input_dataset._as_variant_tensor(),
        self._map_func.function.captured_inputs,
        self._cycle_length,
        self._block_length,
        self._num_parallel_calls,
        f=self._map_func.function,
        **flat_structure(structure=self._output_structure))

  def _transformation_name(self):
    return "Dataset.interleave()"


class FilterDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self, input_dataset, predicate):
    """See `Dataset.filter()` for details."""
    super(FilterDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    wrapped_func = StructuredFunctionWrapper(
        predicate, self._transformation_name(), dataset=input_dataset)
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

  def _transformation_name(self):
    return "Dataset.filter()"


class PrefetchDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size):
    """See `Dataset.prefetch()` for details."""
    super(PrefetchDataset, self).__init__(input_dataset)
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


class WindowDataset(UnaryDataset):
  """A dataset that creates window datasets from the input elements."""

  def __init__(self, input_dataset, size, shift, stride, drop_remainder):
    """See `window_dataset()` for more details."""
    super(WindowDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name="size")
    self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name="shift")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._output_classes = nest.pack_sequence_as(
        input_dataset.output_classes,
        [
            DatasetStructure(
                structure_lib.Structure._from_legacy_structure(  # pylint: disable=protected-access
                    output_type, output_shape, output_class))
            for output_class, output_shape, output_type in zip(
                nest.flatten(input_dataset.output_classes),
                nest.flatten(input_dataset.output_shapes),
                nest.flatten(input_dataset.output_types))
        ])
    self._output_shapes = self._output_classes
    self._output_types = self._output_classes

  def _as_variant_tensor(self):
    return gen_dataset_ops.window_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._size,
        self._shift,
        self._stride,
        self._drop_remainder,
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


class _OptionsDataset(UnaryUnchangedStructureDataset):
  """An identity `Dataset` that stores options."""

  def __init__(self, input_dataset, options):
    super(_OptionsDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._options = input_dataset.options()
    if self._options:
      self._options = self._options.merge(options)
    else:
      self._options = options

  def _as_variant_tensor(self):
    return self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access

  def options(self):
    return self._options


class _ModelDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and models performance."""

  def __init__(self, input_dataset):
    """See `optimize()` for details."""
    super(_ModelDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset

  def _as_variant_tensor(self):
    return gen_dataset_ops.model_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        **flat_structure(self))


class _OptimizeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and applies optimizations."""

  def __init__(self, input_dataset, optimizations):
    """See `optimize()` for details."""
    super(_OptimizeDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    if optimizations is None:
      optimizations = []
    self._optimizations = ops.convert_to_tensor(
        optimizations, dtype=dtypes.string, name="optimizations")

  def _as_variant_tensor(self):
    return gen_dataset_ops.optimize_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._optimizations,
        **flat_structure(self))


class _SetStatsAggregatorDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and sets a stats aggregator."""

  def __init__(self, input_dataset, aggregator, prefix, counter_prefix):
    super(_SetStatsAggregatorDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._stats_aggregator = aggregator
    self._prefix = prefix
    self._counter_prefix = counter_prefix

  def _as_variant_tensor(self):
    return ged_ops.experimental_set_stats_aggregator_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._stats_aggregator._resource,  # pylint: disable=protected-access
        self._prefix,
        self._counter_prefix,
        **flat_structure(self))


class _MaxIntraOpParallelismDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, overriding intra-op parallelism."""

  def __init__(self, input_dataset, max_intra_op_parallelism):
    super(_MaxIntraOpParallelismDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._max_intra_op_parallelism = ops.convert_to_tensor(
        max_intra_op_parallelism,
        dtype=dtypes.int64,
        name="max_intra_op_parallelism")

  def _as_variant_tensor(self):
    return ged_ops.experimental_max_intra_op_parallelism_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._max_intra_op_parallelism,
        **flat_structure(self))


class _PrivateThreadPoolDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, setting a private threadpool."""

  def __init__(self, input_dataset, num_threads):
    super(_PrivateThreadPoolDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset
    self._num_threads = ops.convert_to_tensor(
        num_threads, dtype=dtypes.int64, name="num_threads")

  def _as_variant_tensor(self):
    return ged_ops.experimental_private_thread_pool_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._num_threads,
        **flat_structure(self))
