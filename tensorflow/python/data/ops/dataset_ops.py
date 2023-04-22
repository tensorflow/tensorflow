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
import multiprocessing
import sys
import threading
import warnings
import weakref

from absl import logging
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.training.tracking import base as tracking_base
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency (roughly
# tf.function->wrap_function->dataset->autograph->tf.function).
# TODO(b/133251390): Use a regular import.
wrap_function = lazy_loader.LazyLoader(
    "wrap_function", globals(),
    "tensorflow.python.eager.wrap_function")
# TODO(mdan): Create a public API for this.
autograph_ctx = lazy_loader.LazyLoader(
    "autograph_ctx", globals(),
    "tensorflow.python.autograph.core.ag_ctx")
autograph = lazy_loader.LazyLoader(
    "autograph", globals(),
    "tensorflow.python.autograph.impl.api")

ops.NotDifferentiable("ReduceDataset")

# A constant that can be used to enable auto-tuning.
AUTOTUNE = -1
tf_export("data.AUTOTUNE").export_constant(__name__, "AUTOTUNE")
# TODO(b/168128531): Deprecate and remove this symbol.
tf_export("data.experimental.AUTOTUNE").export_constant(__name__, "AUTOTUNE")

# Constants representing infinite and unknown cardinalities.
INFINITE = -1
UNKNOWN = -2
tf_export("data.INFINITE_CARDINALITY").export_constant(__name__, "INFINITE")
tf_export("data.UNKNOWN_CARDINALITY").export_constant(__name__, "UNKNOWN")


@tf_export("data.Dataset", v1=[])
@six.add_metaclass(abc.ABCMeta)
class DatasetV2(collections_abc.Iterable, tracking_base.Trackable,
                composite_tensor.CompositeTensor):
  """Represents a potentially large set of elements.

  The `tf.data.Dataset` API supports writing descriptive and efficient input
  pipelines. `Dataset` usage follows a common pattern:

  1. Create a source dataset from your input data.
  2. Apply dataset transformations to preprocess the data.
  3. Iterate over the dataset and process the elements.

  Iteration happens in a streaming fashion, so the full dataset does not need to
  fit into memory.

  Source Datasets:

  The simplest way to create a dataset is to create it from a python `list`:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)

  To process lines from files, use `tf.data.TextLineDataset`:

  >>> dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])

  To process records written in the `TFRecord` format, use `TFRecordDataset`:

  >>> dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"])

  To create a dataset of all files matching a pattern, use
  `tf.data.Dataset.list_files`:

  ```python
  dataset = tf.data.Dataset.list_files("/path/*.txt")
  ```

  See `tf.data.FixedLengthRecordDataset` and `tf.data.Dataset.from_generator`
  for more ways to create datasets.

  Transformations:

  Once you have a dataset, you can apply transformations to prepare the data for
  your model:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.map(lambda x: x*2)
  >>> list(dataset.as_numpy_iterator())
  [2, 4, 6]

  Common Terms:

  **Element**: A single output from calling `next()` on a dataset iterator.
    Elements may be nested structures containing multiple components. For
    example, the element `(1, (3, "apple"))` has one tuple nested in another
    tuple. The components are `1`, `3`, and `"apple"`.

  **Component**: The leaf in the nested structure of an element.

  Supported types:

  Elements can be nested structures of tuples, named tuples, and dictionaries.
  Note that Python lists are *not* treated as nested structures of components.
  Instead, lists are converted to tensors and treated as components. For
  example, the element `(1, [1, 2, 3])` has only two components; the tensor `1`
  and the tensor `[1, 2, 3]`. Element components can be of any type
  representable by `tf.TypeSpec`, including `tf.Tensor`, `tf.data.Dataset`,
  `tf.sparse.SparseTensor`, `tf.RaggedTensor`, and `tf.TensorArray`.

  ```python
  a = 1 # Integer element
  b = 2.0 # Float element
  c = (1, 2) # Tuple element with 2 components
  d = {"a": (2, 2), "b": 3} # Dict element with 3 components
  Point = collections.namedtuple("Point", ["x", "y"])
  e = Point(1, 2) # Named tuple
  f = tf.data.Dataset.range(10) # Dataset element
  ```

  For more information,
  read [this guide](https://www.tensorflow.org/guide/data).
  """

  def __init__(self, variant_tensor):
    """Creates a DatasetV2 object.

    This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not
    take anything in its constructor whereas in the DatasetV2, we expect
    subclasses to create a variant_tensor and pass it in to the super() call.

    Args:
      variant_tensor: A DT_VARIANT tensor that represents the dataset.
    """
    self._variant_tensor_attr = variant_tensor
    weak_self = weakref.proxy(self)
    self._variant_tracker = self._track_trackable(
        _VariantTracker(
            self._variant_tensor,
            # _trace_variant_creation only works when executing eagerly, so we
            # don't want to run it immediately. We also want the _VariantTracker
            # to have a weak reference to the Dataset to avoid creating
            # reference cycles and making work for the garbage collector.
            lambda: weak_self._trace_variant_creation()()),  # pylint: disable=unnecessary-lambda,protected-access
        name="_variant_tracker")
    self._graph_attr = ops.get_default_graph()

    # Initialize the options for this dataset and its inputs.
    self._options_attr = Options()
    for input_dataset in self._inputs():
      input_options = None
      if isinstance(input_dataset, DatasetV1):
        # If the V1 dataset does not have the `_dataset` attribute, we assume it
        # is a dataset source and hence does not have options. Otherwise, we
        # grab the options of `_dataset` object
        if hasattr(input_dataset, "_dataset"):
          if not isinstance(input_dataset._dataset, DatasetV2):
            raise AssertionError(
                "The input_dataset._dataset of dataset %s should be DatasetV2."
                % type(self))
          input_options = input_dataset._dataset._options_attr
      elif isinstance(input_dataset, DatasetV2):
        input_options = input_dataset._options_attr
      else:
        raise TypeError("Unexpected dataset type: ", type(input_dataset))
      if input_options is not None:
        self._options_attr = self._options_attr.merge(input_options)
    self._options_attr._set_mutable(False)  # pylint: disable=protected-access

  @property
  def _variant_tensor(self):
    return self._variant_tensor_attr

  @_variant_tensor.setter
  def _variant_tensor(self, _):
    raise ValueError("The _variant_tensor property is read-only")

  @deprecation.deprecated_args(None, "Use external_state_policy instead",
                               "allow_stateful")
  def _as_serialized_graph(
      self,
      allow_stateful=None,
      strip_device_assignment=None,
      external_state_policy=distribute_options.ExternalStatePolicy.WARN):
    """Produces serialized graph representation of the dataset.

    Args:
      allow_stateful: If true, we allow stateful ops to be present in the graph
        def. In that case, the state in these ops would be thrown away.
      strip_device_assignment: If true, non-local (i.e. job and task) device
        assignment is stripped from ops in the serialized graph.
      external_state_policy: The ExternalStatePolicy enum that determines how we
        handle input pipelines that depend on external state. By default, its
        set to WARN.

    Returns:
      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a
      serialized graph.
    """
    if external_state_policy:
      policy = external_state_policy.value
      return gen_dataset_ops.dataset_to_graph_v2(
          self._variant_tensor,
          external_state_policy=policy,
          strip_device_assignment=strip_device_assignment)
    if strip_device_assignment:
      return gen_dataset_ops.dataset_to_graph(
          self._variant_tensor,
          allow_stateful=allow_stateful,
          strip_device_assignment=strip_device_assignment)
    return gen_dataset_ops.dataset_to_graph(
        self._variant_tensor, allow_stateful=allow_stateful)

  def _trace_variant_creation(self):
    """Traces a function which outputs a variant `tf.Tensor` for this dataset.

    Note that creating this function involves evaluating an op, and is currently
    only supported when executing eagerly.

    Returns:
      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.
    """
    variant = self._variant_tensor
    if not isinstance(variant, ops.EagerTensor):
      raise NotImplementedError(
          "Can only export Datasets which were created executing eagerly. "
          "Please file a feature request if this is important to you.")
    with context.eager_mode(), ops.device("CPU"):
      # pylint: disable=protected-access
      graph_def = graph_pb2.GraphDef().FromString(
          self._as_serialized_graph(external_state_policy=distribute_options
                                    .ExternalStatePolicy.FAIL).numpy())
    output_node_name = None
    for node in graph_def.node:
      if node.op == "_Retval":
        if output_node_name is not None:
          raise AssertionError(
              "Found multiple return values from the dataset's graph, expected "
              "only one.")
        output_node_name, = node.input
    if output_node_name is None:
      raise AssertionError("Could not find the dataset's output node.")
    # Add functions used in this Dataset to the function's graph, since they
    # need to follow it around (and for example be added to a SavedModel which
    # references the dataset).
    variant_function = wrap_function.function_from_graph_def(
        graph_def, inputs=[], outputs=output_node_name + ":0")
    for used_function in self._functions():
      used_function.function.add_to_graph(variant_function.graph)
    return variant_function

  @abc.abstractmethod
  def _inputs(self):
    """Returns a list of the input datasets of the dataset."""

    raise NotImplementedError("Dataset._inputs")

  @property
  def _graph(self):
    return self._graph_attr

  @_graph.setter
  def _graph(self, _):
    raise ValueError("The _graph property is read-only")

  # TODO(jsimsa): Change this to be the transitive closure of functions used
  # by this dataset and its inputs.
  def _functions(self):
    """Returns a list of functions associated with this dataset.

    Returns:
      A list of `StructuredFunctionWrapper` objects.
    """
    return []

  def _options(self):
    """Returns the options tensor for this dataset."""
    # pylint: disable=protected-access
    return gen_dataset_ops.get_options(self._variant_tensor)

  @classmethod
  def _options_tensor_to_options(cls, serialized_options):
    """Converts options tensor to tf.data.Options object."""
    options = Options()
    if tensor_util.constant_value(serialized_options) is not None:
      pb = dataset_options_pb2.Options.FromString(tensor_util.constant_value(
          serialized_options))
      options._from_proto(pb)  # pylint: disable=protected-access
    return options

  def options(self):
    """Returns the options for this dataset and its inputs.

    Returns:
      A `tf.data.Options` object representing the dataset options.
    """
    if context.executing_eagerly():
      options = self._options_tensor_to_options(self._options())
      options._set_mutable(False)  # pylint: disable=protected-access
      return options
    warnings.warn("To make it possible to preserve tf.data options across "
                  "serialization boundaries, their implementation has moved to "
                  "be part of the TensorFlow graph. As a consequence, the "
                  "options value is in general no longer known at graph "
                  "construction time. Invoking this method in graph mode "
                  "retains the legacy behavior of the original implementation, "
                  "but note that the returned value might not reflect the "
                  "actual value of the options.")
    return self._options_attr

  def _apply_debug_options(self):
    if DEBUG_MODE:
      # Disable autotuning and static optimizations that could introduce
      # parallelism or asynchrony.
      options = Options()
      options.experimental_optimization.autotune = False
      options.experimental_optimization.map_and_batch_fusion = False
      options.experimental_optimization.map_parallelization = False
      dataset = _OptionsDataset(self, options)
    else:
      dataset = self

    return dataset

  def __iter__(self):
    """Creates an iterator for elements of this dataset.

    The returned iterator implements the Python Iterator protocol.

    Returns:
      An `tf.data.Iterator` for the elements of this dataset.

    Raises:
      RuntimeError: If not inside of tf.function and not executing eagerly.
    """
    if context.executing_eagerly() or ops.inside_function():
      with ops.colocate_with(self._variant_tensor):
        return iterator_ops.OwnedIterator(self)
    else:
      raise RuntimeError("__iter__() is only supported inside of tf.function "
                         "or when eager execution is enabled.")

  def __bool__(self):
    return True  # Required as __len__ is defined

  __nonzero__ = __bool__  # Python 2 backward compatibility

  def __len__(self):
    """Returns the length of the dataset if it is known and finite.

    This method requires that you are running in eager mode, and that the
    length of the dataset is known and non-infinite. When the length may be
    unknown or infinite, or if you are running in graph mode, use
    `tf.data.Dataset.cardinality` instead.

    Returns:
      An integer representing the length of the dataset.

    Raises:
      RuntimeError: If the dataset length is unknown or infinite, or if eager
        execution is not enabled.
    """
    if not context.executing_eagerly():
      raise TypeError("__len__() is not supported while tracing functions. "
                      "Use `tf.data.Dataset.cardinality` instead.")
    length = self.cardinality()
    if length.numpy() == INFINITE:
      raise TypeError("dataset length is infinite.")
    if length.numpy() == UNKNOWN:
      raise TypeError("dataset length is unknown.")
    return length

  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this dataset.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)

    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this dataset and specifying the type of individual components.
    """
    raise NotImplementedError("Dataset.element_spec")

  def __repr__(self):
    output_shapes = nest.map_structure(str, get_legacy_output_shapes(self))
    output_shapes = str(output_shapes).replace("'", "")
    output_types = nest.map_structure(repr, get_legacy_output_types(self))
    output_types = str(output_types).replace("'", "")
    return ("<%s shapes: %s, types: %s>" % (type(self).__name__, output_shapes,
                                            output_types))

  def as_numpy_iterator(self):
    """Returns an iterator which converts all elements of the dataset to numpy.

    Use `as_numpy_iterator` to inspect the content of your dataset. To see
    element shapes and types, print dataset elements directly instead of using
    `as_numpy_iterator`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset:
    ...   print(element)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)

    This method requires that you are running in eager mode and the dataset's
    element_spec contains only `TensorSpec` components.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    1
    2
    3

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> print(list(dataset.as_numpy_iterator()))
    [1, 2, 3]

    `as_numpy_iterator()` will preserve the nested structure of dataset
    elements.

    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),
    ...                                               'b': [5, 6]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},
    ...                                       {'a': (2, 4), 'b': 6}]
    True

    Returns:
      An iterable over the elements of the dataset, with their tensors converted
      to numpy arrays.

    Raises:
      TypeError: if an element contains a non-`Tensor` value.
      RuntimeError: if eager execution is not enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError("as_numpy_iterator() is not supported while tracing "
                         "functions")
    for component_spec in nest.flatten(self.element_spec):
      if not isinstance(
          component_spec,
          (tensor_spec.TensorSpec, ragged_tensor.RaggedTensorSpec)):
        raise TypeError(
            "Dataset.as_numpy_iterator() does not support datasets containing "
            + str(component_spec.value_type))

    return _NumpyIterator(self)

  @property
  def _flat_shapes(self):
    """Returns a list `tf.TensorShapes`s for the element tensor representation.

    Returns:
      A list `tf.TensorShapes`s for the element tensor representation.
    """
    return structure.get_flat_tensor_shapes(self.element_spec)

  @property
  def _flat_types(self):
    """Returns a list `tf.DType`s for the element tensor representation.

    Returns:
      A list `tf.DType`s for the element tensor representation.
    """
    return structure.get_flat_tensor_types(self.element_spec)

  @property
  def _flat_structure(self):
    """Helper for setting `output_shapes` and `output_types` attrs of an op.

    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element. This helper
    function generates these attrs as a keyword argument dictionary, allowing
    `Dataset._variant_tensor` implementations to pass `**self._flat_structure`
    to the op constructor.

    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }

  @property
  def _type_spec(self):
    return DatasetSpec(self.element_spec)

  @staticmethod
  def from_tensors(tensors):
    """Creates a `Dataset` with a single element, comprising the given tensors.

    `from_tensors` produces a dataset containing only a single element. To slice
    the input tensor into multiple elements, use `from_tensor_slices` instead.

    >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32)]
    >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
    >>> list(dataset.as_numpy_iterator())
    [(array([1, 2, 3], dtype=int32), b'A')]

    >>> # You can use `from_tensors` to produce a dataset which repeats
    >>> # the same example many times.
    >>> example = tf.constant([1,2,3])
    >>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).

    Args:
      tensors: A dataset "element". Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorDataset(tensors)

  @staticmethod
  def from_tensor_slices(tensors):
    """Creates a `Dataset` whose elements are slices of the given tensors.

    The given tensors are sliced along their first dimension. This operation
    preserves the structure of the input tensors, removing the first dimension
    of each tensor and using it as the dataset dimension. All input tensors
    must have the same size in their first dimensions.

    >>> # Slicing a 1D tensor produces scalar tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3]

    >>> # Slicing a 2D tensor produces 1D tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]

    >>> # Slicing a tuple of 1D tensors produces tuple elements containing
    >>> # scalar tensors.
    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
    >>> list(dataset.as_numpy_iterator())
    [(1, 3, 5), (2, 4, 6)]

    >>> # Dictionary structure is also preserved.
    >>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
    ...                                       {'a': 2, 'b': 4}]
    True

    >>> # Two tensors can be combined into one Dataset object.
    >>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
    >>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
    >>> dataset = Dataset.from_tensor_slices((features, labels))
    >>> # Both the features and the labels tensors can be converted
    >>> # to a Dataset object separately and combined after.
    >>> features_dataset = Dataset.from_tensor_slices(features)
    >>> labels_dataset = Dataset.from_tensor_slices(labels)
    >>> dataset = Dataset.zip((features_dataset, labels_dataset))
    >>> # A batched feature and label set can be converted to a Dataset
    >>> # in similar fashion.
    >>> batched_features = tf.constant([[[1, 3], [2, 3]],
    ...                                 [[2, 1], [1, 2]],
    ...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
    >>> batched_labels = tf.constant([['A', 'A'],
    ...                               ['B', 'B'],
    ...                               ['A', 'B']], shape=(3, 2, 1))
    >>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (array([[1, 3],
           [2, 3]], dtype=int32), array([[b'A'],
           [b'A']], dtype=object))
    (array([[2, 1],
           [1, 2]], dtype=int32), array([[b'B'],
           [b'B']], dtype=object))
    (array([[3, 3],
           [3, 2]], dtype=int32), array([[b'A'],
           [b'B']], dtype=object))

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/data#consuming_numpy_arrays).

    Args:
      tensors: A dataset element, whose components have the same first
        dimension. Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).

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
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None):
    """Creates a `Dataset` whose elements are generated by `generator`.

    The `generator` argument must be a callable object that returns
    an object that supports the `iter()` protocol (e.g. a generator function).

    The elements generated by `generator` must be compatible with either the
    given `output_signature` argument or with the given `output_types` and
    (optionally) `output_shapes` arguments, whichever was specified.

    The recommended way to call `from_generator` is to use the
    `output_signature` argument. In this case the output will be assumed to
    consist of objects with the classes, shapes and types defined by
    `tf.TypeSpec` objects from `output_signature` argument:

    >>> def gen():
    ...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    ...   yield 42, ragged_tensor
    >>>
    >>> dataset = tf.data.Dataset.from_generator(
    ...      gen,
    ...      output_signature=(
    ...          tf.TensorSpec(shape=(), dtype=tf.int32),
    ...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
    >>>
    >>> list(dataset.take(1))
    [(<tf.Tensor: shape=(), dtype=int32, numpy=42>,
    <tf.RaggedTensor [[1, 2], [3]]>)]

    There is also a deprecated way to call `from_generator` by either with
    `output_types` argument alone or together with `output_shapes` argument.
    In this case the output of the function will be assumed to consist of
    `tf.Tensor` objects with the types defined by `output_types` and with the
    shapes which are either unknown or defined by `output_shapes`.

    Note: The current implementation of `Dataset.from_generator()` uses
    `tf.numpy_function` and inherits the same constraints. In particular, it
    requires the dataset and iterator related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. The body of `generator` will not be
    serialized in a `GraphDef`, and you should not use this method if you
    need to serialize your model and restore it in a different environment.

    Note: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.

    Note: While the `output_signature` parameter makes it possible to yield
    `Dataset` elements, the scope of `Dataset.from_generator()` should be
    limited to logic that cannot be expressed through tf.data operations. Using
    tf.data operations within the generator function is an anti-pattern and may
    result in incremental memory growth.

    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take no
        arguments; otherwise it must take as many arguments as there are values
        in `args`.
      output_types: (Optional.) A (nested) structure of `tf.DType` objects
        corresponding to each component of an element yielded by `generator`.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element yielded by
        `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.
      output_signature: (Optional.) A (nested) structure of `tf.TypeSpec`
        objects corresponding to each component of an element yielded by
        `generator`.

    Returns:
      Dataset: A `Dataset`.
    """
    if not callable(generator):
      raise TypeError("`generator` must be callable.")

    if output_signature is not None:
      if output_types is not None:
        raise TypeError("`output_types` can not be used together with "
                        "`output_signature`")
      if output_shapes is not None:
        raise TypeError("`output_shapes` can not be used together with "
                        "`output_signature`")
      if not all(
          isinstance(_, type_spec.TypeSpec)
          for _ in nest.flatten(output_signature)):
        raise TypeError("All the elements of `output_signature` must be "
                        "`tf.TypeSpec` objects.")
    else:
      if output_types is None:
        raise TypeError("Either `output_signature` or `output_types` must "
                        "be specified")

    if output_signature is None:
      if output_shapes is None:
        output_shapes = nest.map_structure(
            lambda _: tensor_shape.TensorShape(None), output_types)
      else:
        output_shapes = nest.map_structure_up_to(output_types,
                                                 tensor_shape.as_shape,
                                                 output_shapes)
      output_signature = nest.map_structure_up_to(output_types,
                                                  tensor_spec.TensorSpec,
                                                  output_shapes, output_types)
    if all(
        isinstance(x, tensor_spec.TensorSpec)
        for x in nest.flatten(output_signature)):
      output_types = nest.pack_sequence_as(
          output_signature, [x.dtype for x in nest.flatten(output_signature)])
      output_shapes = nest.pack_sequence_as(
          output_signature, [x.shape for x in nest.flatten(output_signature)])

    if args is None:
      args = ()
    else:
      args = tuple(ops.convert_n_to_tensor(args, name="args"))

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
      return script_ops.numpy_function(generator_state.get_next_id, args,
                                       dtypes.int64)

    def generator_next_fn(iterator_id_t):
      """Generates the next element from iterator with ID `iterator_id_t`.

      We map this function across an infinite repetition of the
      `iterator_id_t`, and raise `StopIteration` to terminate the iteration.

      Args:
        iterator_id_t: A `tf.int64` tensor whose value uniquely identifies the
          iterator in `generator_state` from which to generate an element.

      Returns:
        The next element to generate from the iterator.
      """
      if output_types and output_shapes:
        flattened_types = [
            dtypes.as_dtype(dt) for dt in nest.flatten(output_types)
        ]
        flattened_shapes = nest.flatten(output_shapes)

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
            six.reraise(
                TypeError,
                TypeError(
                    "`generator` yielded an element that did not match the "
                    "expected structure. The expected structure was %s, but "
                    "the yielded element was %s." % (output_types, values)),
                sys.exc_info()[2])
          ret_arrays = []
          for ret, dtype in zip(flattened_values, flattened_types):
            try:
              ret_arrays.append(
                  script_ops.FuncRegistry._convert(  # pylint: disable=protected-access
                      ret,
                      dtype=dtype.as_numpy_dtype))
            except (TypeError, ValueError):
              six.reraise(
                  TypeError,
                  TypeError(
                      "`generator` yielded an element that could not be "
                      "converted to the expected type. The expected type was "
                      "%s, but the yielded element was %s." %
                      (dtype.name, ret)),
                  sys.exc_info()[2])

          # Additional type and shape checking to ensure that the components of
          # the generated element match the `output_types` and `output_shapes`
          # arguments.
          for (ret_array, expected_dtype,
               expected_shape) in zip(ret_arrays, flattened_types,
                                      flattened_shapes):
            if ret_array.dtype != expected_dtype.as_numpy_dtype:
              raise TypeError(
                  "`generator` yielded an element of type %s where an element "
                  "of type %s was expected." %
                  (ret_array.dtype, expected_dtype.as_numpy_dtype))
            if not expected_shape.is_compatible_with(ret_array.shape):
              raise ValueError(
                  "`generator` yielded an element of shape %s where an element "
                  "of shape %s was expected." %
                  (ret_array.shape, expected_shape))

          return ret_arrays

        flat_values = script_ops.numpy_function(generator_py_func,
                                                [iterator_id_t],
                                                flattened_types)

        # The `py_func()` op drops the inferred shapes, so we add them back in
        # here.
        if output_shapes is not None:
          for ret_t, shape in zip(flat_values, flattened_shapes):
            ret_t.set_shape(shape)

        return nest.pack_sequence_as(output_types, flat_values)
      else:
        flat_output_types = structure.get_flat_tensor_types(output_signature)

        def generator_py_func(iterator_id):
          """A `py_func` that will be called to invoke the iterator."""
          # `next()` raises `StopIteration` when there are no more
          # elements remaining to be generated.
          values = next(generator_state.get_iterator(iterator_id.numpy()))

          try:
            values = structure.normalize_element(values, output_signature)
          except (TypeError, ValueError):
            six.reraise(
                TypeError,
                TypeError(
                    "`generator` yielded an element that did not match the "
                    "expected structure. The expected structure was %s, but "
                    "the yielded element was %s." % (output_signature, values)),
                sys.exc_info()[2])

          values_spec = structure.type_spec_from_value(values)

          if not structure.are_compatible(values_spec, output_signature):
            raise TypeError(
                "`generator` yielded an element of %s where an element "
                "of %s was expected." % (values_spec, output_signature))

          return structure.to_tensor_list(output_signature, values)

        return script_ops._eager_py_func(  # pylint: disable=protected-access
            generator_py_func,
            inp=[iterator_id_t],
            Tout=flat_output_types,
            use_tape_cache=False)

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

      return script_ops.numpy_function(finalize_py_func, [iterator_id_t],
                                       dtypes.int64)

    # This function associates each traversal of `generator` with a unique
    # iterator ID.
    def flat_map_fn(dummy_arg):
      # The `get_iterator_id_fn` gets a unique ID for the current instance of
      # of the generator.
      # The `generator_next_fn` gets the next element from the iterator with the
      # given ID, and raises StopIteration when that iterator contains no
      # more elements.
      return _GeneratorDataset(dummy_arg, get_iterator_id_fn, generator_next_fn,
                               finalize_fn, output_signature)

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
  def range(*args, **kwargs):
    """Creates a `Dataset` of a step-separated range of values.

    >>> list(Dataset.range(5).as_numpy_iterator())
    [0, 1, 2, 3, 4]
    >>> list(Dataset.range(2, 5).as_numpy_iterator())
    [2, 3, 4]
    >>> list(Dataset.range(1, 5, 2).as_numpy_iterator())
    [1, 3]
    >>> list(Dataset.range(1, 5, -2).as_numpy_iterator())
    []
    >>> list(Dataset.range(5, 1).as_numpy_iterator())
    []
    >>> list(Dataset.range(5, 1, -2).as_numpy_iterator())
    [5, 3]
    >>> list(Dataset.range(2, 5, output_type=tf.int32).as_numpy_iterator())
    [2, 3, 4]
    >>> list(Dataset.range(1, 5, 2, output_type=tf.float32).as_numpy_iterator())
    [1.0, 3.0]

    Args:
      *args: follows the same semantics as python's xrange.
        len(args) == 1 -> start = 0, stop = args[0], step = 1.
        len(args) == 2 -> start = args[0], stop = args[1], step = 1.
        len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
      **kwargs:
        - output_type: Its expected dtype. (Optional, default: `tf.int64`).

    Returns:
      Dataset: A `RangeDataset`.

    Raises:
      ValueError: if len(args) == 0.
    """
    return RangeDataset(*args, **kwargs)

  @staticmethod
  def zip(datasets):
    """Creates a `Dataset` by zipping together the given datasets.

    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be a (nested) structure of `Dataset` objects. The supported
    nesting mechanisms are documented
    [here] (https://www.tensorflow.org/guide/data#dataset_structure).

    >>> # The nested structure of the `datasets` argument determines the
    >>> # structure of elements in the resulting dataset.
    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
    >>> ds = tf.data.Dataset.zip((a, b))
    >>> list(ds.as_numpy_iterator())
    [(1, 4), (2, 5), (3, 6)]
    >>> ds = tf.data.Dataset.zip((b, a))
    >>> list(ds.as_numpy_iterator())
    [(4, 1), (5, 2), (6, 3)]
    >>>
    >>> # The `datasets` argument may contain an arbitrary number of datasets.
    >>> c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],
    ...                                            #       [9, 10],
    ...                                            #       [11, 12] ]
    >>> ds = tf.data.Dataset.zip((a, b, c))
    >>> for element in ds.as_numpy_iterator():
    ...   print(element)
    (1, 4, array([7, 8]))
    (2, 5, array([ 9, 10]))
    (3, 6, array([11, 12]))
    >>>
    >>> # The number of elements in the resulting dataset is the same as
    >>> # the size of the smallest dataset in `datasets`.
    >>> d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]
    >>> ds = tf.data.Dataset.zip((a, d))
    >>> list(ds.as_numpy_iterator())
    [(1, 13), (2, 14)]

    Args:
      datasets: A (nested) structure of datasets.

    Returns:
      Dataset: A `Dataset`.
    """
    return ZipDataset(datasets)

  def concatenate(self, dataset):
    """Creates a `Dataset` by concatenating the given dataset with this dataset.

    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
    >>> ds = a.concatenate(b)
    >>> list(ds.as_numpy_iterator())
    [1, 2, 3, 4, 5, 6, 7]
    >>> # The input dataset and dataset to be concatenated should have
    >>> # compatible element specs.
    >>> c = tf.data.Dataset.zip((a, b))
    >>> a.concatenate(c)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and (tf.int64, tf.int64)
    >>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    >>> a.concatenate(d)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and <dtype: 'string'>

    Args:
      dataset: `Dataset` to be concatenated.

    Returns:
      Dataset: A `Dataset`.
    """
    return ConcatenateDataset(self, dataset)

  def prefetch(self, buffer_size):
    """Creates a `Dataset` that prefetches elements from this dataset.

    Most dataset input pipelines should end with a call to `prefetch`. This
    allows later elements to be prepared while the current element is being
    processed. This often improves latency and throughput, at the cost of
    using additional memory to store prefetched elements.

    Note: Like other `Dataset` methods, prefetch operates on the
    elements of the input dataset. It has no concept of examples vs. batches.
    `examples.prefetch(2)` will prefetch two elements (2 examples),
    while `examples.batch(20).prefetch(2)` will prefetch 2 elements
    (2 batches, of 20 examples each).

    >>> dataset = tf.data.Dataset.range(3)
    >>> dataset = dataset.prefetch(2)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2]

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
        number of elements that will be buffered when prefetching. If the value
        `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
    Returns:
      Dataset: A `Dataset`.
    """
    if DEBUG_MODE:
      return self
    return PrefetchDataset(self, buffer_size)

  @staticmethod
  def list_files(file_pattern, shuffle=None, seed=None):
    """A dataset of all files matching one or more glob patterns.

    The `file_pattern` argument should be a small number of glob patterns.
    If your filenames have already been globbed, use
    `Dataset.from_tensor_slices(filenames)` instead, as re-globbing every
    filename with `list_files` may result in poor performance with remote
    storage systems.

    Note: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.

    Example:
      If we had the following files on our filesystem:

        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py

      If we pass "/path/to/dir/*.py" as the directory, the dataset
      would produce:

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
        `tf.random.set_seed` for behavior.

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
    """Repeats this dataset so each original value is seen `count` times.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.repeat(3)
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3, 1, 2, 3, 1, 2, 3]

    Note: If this dataset is a function of global state (e.g. a random number
    generator), then different repetitions may produce different elements.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior (if
        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.

    Returns:
      Dataset: A `Dataset`.
    """
    return RepeatDataset(self, count)

  def enumerate(self, start=0):
    """Enumerates the elements of this dataset.

    It is similar to python's `enumerate`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.enumerate(start=5)
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (5, 1)
    (6, 2)
    (7, 3)

    >>> # The (nested) structure of the input dataset determines the
    >>> # structure of elements in the resulting dataset.
    >>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
    >>> dataset = dataset.enumerate()
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (0, array([7, 8], dtype=int32))
    (1, array([ 9, 10], dtype=int32))

    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
        enumeration.

    Returns:
      Dataset: A `Dataset`.
    """

    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    return Dataset.zip((Dataset.range(start, max_value), self))

  def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset.

    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.

    For instance, if your dataset contains 10,000 elements but `buffer_size` is
    set to 1,000, then `shuffle` will initially select a random element from
    only the first 1,000 elements in the buffer. Once an element is selected,
    its space in the buffer is replaced by the next (i.e. 1,001-st) element,
    maintaining the 1,000 element buffer.

    `reshuffle_each_iteration` controls whether the shuffle order should be
    different for each epoch. In TF 1.X, the idiomatic way to create epochs
    was through the `repeat` transformation:

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 2, 0]

    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 0, 2]
    ```

    In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
    possible to also create epochs through Python iteration:

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 2, 0]
    ```

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    ```

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      Dataset: A `Dataset`.
    """
    return ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)

  def cache(self, filename=""):
    """Caches the elements in this dataset.

    The first time the dataset is iterated over, its elements will be cached
    either in the specified file or in memory. Subsequent iterations will
    use the cached data.

    Note: For the cache to be finalized, the input dataset must be iterated
    through in its entirety. Otherwise, subsequent iterations will not use
    cached data.

    >>> dataset = tf.data.Dataset.range(5)
    >>> dataset = dataset.map(lambda x: x**2)
    >>> dataset = dataset.cache()
    >>> # The first time reading through the data will generate the data using
    >>> # `range` and `map`.
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 4, 9, 16]
    >>> # Subsequent iterations read from the cache.
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 4, 9, 16]

    When caching to a file, the cached data will persist across runs. Even the
    first iteration through the data will read from the cache file. Changing
    the input pipeline before the call to `.cache()` will have no effect until
    the cache file is removed or the filename is changed.

    ```python
    dataset = tf.data.Dataset.range(5)
    dataset = dataset.cache("/path/to/file")
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.cache("/path/to/file")  # Same file!
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    ```

    Note: `cache` will produce exactly the same elements during each iteration
    through the dataset. If you wish to randomize the iteration order, make sure
    to call `shuffle` *after* calling `cache`.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching elements in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.

    Returns:
      Dataset: A `Dataset`.
    """
    return CacheDataset(self, filename)

  def take(self, count):
    """Creates a `Dataset` with at most `count` elements from this dataset.

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.take(3)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2]

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

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.skip(7)
    >>> list(dataset.as_numpy_iterator())
    [7, 8, 9]

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be skipped to form the new dataset.
        If `count` is greater than the size of this dataset, the new dataset
        will contain no elements.  If `count` is -1, skips the entire dataset.

    Returns:
      Dataset: A `Dataset`.
    """
    return SkipDataset(self, count)

  def shard(self, num_shards, index):
    """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

    `shard` is deterministic. The Dataset produced by `A.shard(n, i)` will
    contain all elements of A whose index mod n = i.

    >>> A = tf.data.Dataset.range(10)
    >>> B = A.shard(num_shards=3, index=0)
    >>> list(B.as_numpy_iterator())
    [0, 3, 6, 9]
    >>> C = A.shard(num_shards=3, index=1)
    >>> list(C.as_numpy_iterator())
    [1, 4, 7]
    >>> D = A.shard(num_shards=3, index=2)
    >>> list(D.as_numpy_iterator())
    [2, 5, 8]

    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.

    When reading a single input file, you can shard elements as follows:

    ```python
    d = tf.data.TFRecordDataset(input_file)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
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
    d = Dataset.list_files(pattern)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

    Returns:
      Dataset: A `Dataset`.

    Raises:
      InvalidArgumentError: if `num_shards` or `index` are illegal values.

        Note: error checking is done on a best-effort basis, and errors aren't
        guaranteed to be caught upon dataset creation. (e.g. providing in a
        placeholder tensor bypasses the early checking, and will instead result
        in an error during a session.run call.)
    """
    return ShardDataset(self, num_shards, index)

  def batch(self,
            batch_size,
            drop_remainder=False,
            num_parallel_calls=None,
            deterministic=None):
    """Combines consecutive elements of this dataset into batches.

    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]

    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3, drop_remainder=True)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]

    The components of the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Note: If your program requires data to have a statically known shape (e.g.,
    when using XLA), you should use `drop_remainder=True`. Without
    `drop_remainder=True` the shape of the output dataset will have an unknown
    leading dimension due to the possibility of a smaller final batch.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number of batches to compute asynchronously in
        parallel.
        If not specified, batches will be computed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available resources.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.experimental_deterministic` option
        (`True` by default) controls the behavior.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None or DEBUG_MODE:
      if deterministic is not None and not DEBUG_MODE:
        warnings.warn("The `deterministic` argument has no effect unless the "
                      "`num_parallel_calls` argument is specified.")
      return BatchDataset(self, batch_size, drop_remainder)
    else:
      return ParallelBatchDataset(self, batch_size, drop_remainder,
                                  num_parallel_calls, deterministic)

  def padded_batch(self,
                   batch_size,
                   padded_shapes=None,
                   padding_values=None,
                   drop_remainder=False):
    """Combines consecutive elements of this dataset into padded batches.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element.

    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padded_shapes`. The `padded_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant, the component will be padded out to that
      length in that dimension.
    * If the dimension is unknown, the component will be padded out to the
      maximum length of all elements in that dimension.

    >>> A = (tf.data.Dataset
    ...      .range(1, 5, output_type=tf.int32)
    ...      .map(lambda x: tf.fill([x], x)))
    >>> # Pad to the smallest per-batch size that fits all elements.
    >>> B = A.padded_batch(2)
    >>> for element in B.as_numpy_iterator():
    ...   print(element)
    [[1 0]
     [2 2]]
    [[3 3 3 0]
     [4 4 4 4]]
    >>> # Pad to a fixed size.
    >>> C = A.padded_batch(2, padded_shapes=5)
    >>> for element in C.as_numpy_iterator():
    ...   print(element)
    [[1 0 0 0 0]
     [2 2 0 0 0]]
    [[3 3 3 0 0]
     [4 4 4 4 0]]
    >>> # Pad with a custom value.
    >>> D = A.padded_batch(2, padded_shapes=5, padding_values=-1)
    >>> for element in D.as_numpy_iterator():
    ...   print(element)
    [[ 1 -1 -1 -1 -1]
     [ 2  2 -1 -1 -1]]
    [[ 3  3  3 -1 -1]
     [ 4  4  4  4 -1]]
    >>> # Components of nested elements can be padded independently.
    >>> elements = [([1, 2, 3], [10]),
    ...             ([4, 5], [11, 12])]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: iter(elements), (tf.int32, tf.int32))
    >>> # Pad the first component of the tuple to length 4, and the second
    >>> # component to the smallest size that fits.
    >>> dataset = dataset.padded_batch(2,
    ...     padded_shapes=([4], [None]),
    ...     padding_values=(-1, 100))
    >>> list(dataset.as_numpy_iterator())
    [(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),
      array([[ 10, 100], [ 11,  12]], dtype=int32))]
    >>> # Pad with a single value and multiple components.
    >>> E = tf.data.Dataset.zip((A, A)).padded_batch(2, padding_values=-1)
    >>> for element in E.as_numpy_iterator():
    ...   print(element)
    (array([[ 1, -1],
           [ 2,  2]], dtype=int32), array([[ 1, -1],
           [ 2,  2]], dtype=int32))
    (array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32), array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32))

    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.sparse.SparseTensor`.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or
        `tf.int64` vector tensor-like objects representing the shape to which
        the respective component of each input element should be padded prior
        to batching. Any unknown dimensions will be padded to the maximum size
        of that dimension in each batch. If unset, all dimensions of all
        components are padded to the maximum size in the batch. `padded_shapes`
        must be set if any component has an unknown rank.
      padding_values: (Optional.) A (nested) structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the respective
        components. None represents that the (nested) structure should be padded
        with default values.  Defaults are `0` for numeric types and the empty
        string for string types. The `padding_values` should have the same
        (nested) structure as the input dataset. If `padding_values` is a single
        element and the input dataset has multiple components, then the same
        `padding_values` will be used to pad every component of the dataset.
        If `padding_values` is a scalar, then its value will be broadcasted
        to match the shape of each component.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.

    Raises:
      ValueError: If a component has an unknown rank, and  the `padded_shapes`
        argument is not set.
    """
    if padded_shapes is None:
      padded_shapes = get_legacy_output_shapes(self)
      # A `tf.TensorShape` is only false if its *rank* is unknown:
      # bool(tf.TensorShape(None)) is False
      if not all(nest.flatten(padded_shapes)):
        raise ValueError("You must set the `padded_shapes` argument to "
                         "`Dataset.padded_batch` if any component of its "
                         "input has an unknown rank")
    return PaddedBatchDataset(self, batch_size, padded_shapes, padding_values,
                              drop_remainder)

  def map(self, map_func, num_parallel_calls=None, deterministic=None):
    """Maps `map_func` across the elements of this dataset.

    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input. `map_func` can be used to change both
    the values and the structure of a dataset's elements. Supported structure
    constructs are documented
    [here](https://www.tensorflow.org/guide/data#dataset_structure).

    For example, `map` can be used for adding 1 to each element, or projecting a
    subset of element components.

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> dataset = dataset.map(lambda x: x + 1)
    >>> list(dataset.as_numpy_iterator())
    [2, 3, 4, 5, 6]

    The input signature of `map_func` is determined by the structure of each
    element in this dataset.

    >>> dataset = Dataset.range(5)
    >>> # `map_func` takes a single argument of type `tf.Tensor` with the same
    >>> # shape and dtype.
    >>> result = dataset.map(lambda x: x + 1)

    >>> # Each element is a tuple containing two `tf.Tensor` objects.
    >>> elements = [(1, "foo"), (2, "bar"), (3, "baz")]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, (tf.int32, tf.string))
    >>> # `map_func` takes two arguments of type `tf.Tensor`. This function
    >>> # projects out just the first component.
    >>> result = dataset.map(lambda x_int, y_str: x_int)
    >>> list(result.as_numpy_iterator())
    [1, 2, 3]

    >>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    >>> elements =  ([{"a": 1, "b": "foo"},
    ...               {"a": 2, "b": "bar"},
    ...               {"a": 3, "b": "baz"}])
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, {"a": tf.int32, "b": tf.string})
    >>> # `map_func` takes a single argument of type `dict` with the same keys
    >>> # as the elements.
    >>> result = dataset.map(lambda d: str(d["a"]) + d["b"])

    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.

    >>> dataset = tf.data.Dataset.range(3)
    >>> # `map_func` returns two `tf.Tensor` objects.
    >>> def g(x):
    ...   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
    >>> result = dataset.map(g)
    >>> result.element_spec
    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(3,), \
dtype=tf.string, name=None))
    >>> # Python primitives, lists, and NumPy arrays are implicitly converted to
    >>> # `tf.Tensor`.
    >>> def h(x):
    ...   return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)
    >>> result = dataset.map(h)
    >>> result.element_spec
    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(2,), \
dtype=tf.string, name=None), TensorSpec(shape=(2,), dtype=tf.float64, \
name=None))
    >>> # `map_func` can return nested structures.
    >>> def i(x):
    ...   return (37.0, [42, 16]), "foo"
    >>> result = dataset.map(i)
    >>> result.element_spec
    ((TensorSpec(shape=(), dtype=tf.float32, name=None),
      TensorSpec(shape=(2,), dtype=tf.int32, name=None)),
     TensorSpec(shape=(), dtype=tf.string, name=None))

    `map_func` can accept as arguments and return any type of dataset element.

    Note that irrespective of the context in which `map_func` is defined (eager
    vs. graph), tf.data traces the function and executes it as a graph. To use
    Python code inside of the function you have a few options:

    1) Rely on AutoGraph to convert Python code into an equivalent graph
    computation. The downside of this approach is that AutoGraph can convert
    some but not all Python code.

    2) Use `tf.py_function`, which allows you to write arbitrary Python code but
    will generally result in worse performance than 1). For example:

    >>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    >>> # transform a string tensor to upper case string using a Python function
    >>> def upper_case_fn(t: tf.Tensor):
    ...   return t.numpy().decode('utf-8').upper()
    >>> d = d.map(lambda x: tf.py_function(func=upper_case_fn,
    ...           inp=[x], Tout=tf.string))
    >>> list(d.as_numpy_iterator())
    [b'HELLO', b'WORLD']

    3) Use `tf.numpy_function`, which also allows you to write arbitrary
    Python code. Note that `tf.py_function` accepts `tf.Tensor` whereas
    `tf.numpy_function` accepts numpy arrays and returns only numpy arrays.
    For example:

    >>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    >>> def upper_case_fn(t: np.ndarray):
    ...   return t.decode('utf-8').upper()
    >>> d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,
    ...           inp=[x], Tout=tf.string))
    >>> list(d.as_numpy_iterator())
    [b'HELLO', b'WORLD']

    Note that the use of `tf.numpy_function` and `tf.py_function`
    in general precludes the possibility of executing user-defined
    transformations in parallel (because of Python GIL).

    Performance can often be improved by setting `num_parallel_calls` so that
    `map` will use multiple threads to process elements. If deterministic order
    isn't required, it can also improve performance to set
    `deterministic=False`.

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> dataset = dataset.map(lambda x: x + 1,
    ...     num_parallel_calls=tf.data.AUTOTUNE,
    ...     deterministic=False)

    The order of elements yielded by this transformation is deterministic if
    `deterministic=True`. If `map_func` contains stateful operations and
    `num_parallel_calls > 1`, the order in which that state is accessed is
    undefined, so the values of output elements may not be deterministic
    regardless of the `deterministic` flag value.

    Args:
      map_func: A function mapping a dataset element to another dataset element.
      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.experimental_deterministic` option
        (`True` by default) controls the behavior.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None or DEBUG_MODE:
      if deterministic is not None and not DEBUG_MODE:
        warnings.warn("The `deterministic` argument has no effect unless the "
                      "`num_parallel_calls` argument is specified.")
      return MapDataset(self, map_func, preserve_cardinality=True)
    else:
      return ParallelMapDataset(
          self,
          map_func,
          num_parallel_calls,
          deterministic,
          preserve_cardinality=True)

  def flat_map(self, map_func):
    """Maps `map_func` across this dataset and flattens the result.

    The type signature is:

    ```
    def flat_map(
      self: Dataset[T],
      map_func: Callable[[T], Dataset[S]]
    ) -> Dataset[S]
    ```

    Use `flat_map` if you want to make sure that the order of your dataset
    stays the same. For example, to flatten a dataset of batches into a
    dataset of their elements:

    >>> dataset = tf.data.Dataset.from_tensor_slices(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> dataset = dataset.flat_map(
    ...     lambda x: tf.data.Dataset.from_tensor_slices(x))
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
    `flat_map` produces the same output as
    `tf.data.Dataset.interleave(cycle_length=1)`

    Args:
      map_func: A function mapping a dataset element to a dataset.

    Returns:
      Dataset: A `Dataset`.
    """
    return FlatMapDataset(self, map_func)

  def interleave(self,
                 map_func,
                 cycle_length=None,
                 block_length=None,
                 num_parallel_calls=None,
                 deterministic=None):
    """Maps `map_func` across this dataset, and interleaves the results.

    The type signature is:

    ```
    def interleave(
      self: Dataset[T],
      map_func: Callable[[T], Dataset[S]]
    ) -> Dataset[S]
    ```

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    >>> # Preprocess 4 files concurrently, and interleave blocks of 16 records
    >>> # from each file.
    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
    ...              "/var/data/file3.txt", "/var/data/file4.txt"]
    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
    >>> def parse_fn(filename):
    ...   return tf.data.Dataset.range(10)
    >>> dataset = dataset.interleave(lambda x:
    ...     tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
    ...     cycle_length=4, block_length=16)

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

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> # NOTE: New lines indicate "block" boundaries.
    >>> dataset = dataset.interleave(
    ...     lambda x: Dataset.from_tensors(x).repeat(6),
    ...     cycle_length=2, block_length=4)
    >>> list(dataset.as_numpy_iterator())
    [1, 1, 1, 1,
     2, 2, 2, 2,
     1, 1,
     2, 2,
     3, 3, 3, 3,
     4, 4, 4, 4,
     3, 3,
     4, 4,
     5, 5, 5, 5,
     5, 5]

    Note: The order of elements yielded by this transformation is
    deterministic, as long as `map_func` is a pure function and
    `deterministic=True`. If `map_func` contains any stateful operations, the
    order in which that state is accessed is undefined.

    Performance can often be improved by setting `num_parallel_calls` so that
    `interleave` will use multiple threads to fetch elements. If determinism
    isn't required, it can also improve performance to set
    `deterministic=False`.

    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
    ...              "/var/data/file3.txt", "/var/data/file4.txt"]
    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
    >>> dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    ...     cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE,
    ...     deterministic=False)

    Args:
      map_func: A function that takes a dataset element and returns a
        `tf.data.Dataset`.
      cycle_length: (Optional.) The number of input elements that will be
        processed concurrently. If not set, the tf.data runtime decides what it
        should be based on available CPU. If `num_parallel_calls` is set to
        `tf.data.AUTOTUNE`, the `cycle_length` argument identifies
        the maximum degree of parallelism.
      block_length: (Optional.) The number of consecutive elements to produce
        from each input element before cycling to another input element. If not
        set, defaults to 1.
      num_parallel_calls: (Optional.) If specified, the implementation creates a
        threadpool, which is used to fetch inputs from cycle elements
        asynchronously and in parallel. The default behavior is to fetch inputs
        from cycle elements synchronously with no parallelism. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.experimental_deterministic` option
        (`True` by default) controls the behavior.

    Returns:
      Dataset: A `Dataset`.
    """
    if block_length is None:
      block_length = 1

    if cycle_length is None:
      cycle_length = AUTOTUNE

    if num_parallel_calls is None or DEBUG_MODE:
      if deterministic is not None and not DEBUG_MODE:
        warnings.warn("The `deterministic` argument has no effect unless the "
                      "`num_parallel_calls` argument is specified.")
      return InterleaveDataset(self, map_func, cycle_length, block_length)
    else:
      return ParallelInterleaveDataset(
          self,
          map_func,
          cycle_length,
          block_length,
          num_parallel_calls,
          deterministic=deterministic)

  def filter(self, predicate):
    """Filters this dataset according to `predicate`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.filter(lambda x: x < 3)
    >>> list(dataset.as_numpy_iterator())
    [1, 2]
    >>> # `tf.math.equal(x, y)` is required for equality comparison
    >>> def filter_fn(x):
    ...   return tf.math.equal(x, 1)
    >>> dataset = dataset.filter(filter_fn)
    >>> list(dataset.as_numpy_iterator())
    [1]

    Args:
      predicate: A function mapping a dataset element to a boolean.

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

    >>> dataset = tf.data.Dataset.range(100)
    >>> def dataset_fn(ds):
    ...   return ds.filter(lambda x: x < 5)
    >>> dataset = dataset.apply(dataset_fn)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2, 3, 4]

    Args:
      transformation_func: A function that takes one `Dataset` argument and
        returns a `Dataset`.

    Returns:
      Dataset: The `Dataset` returned by applying `transformation_func` to this
          dataset.
    """
    dataset = transformation_func(self)
    if not isinstance(dataset, DatasetV2):
      raise TypeError(
          "`transformation_func` must return a Dataset. Got {}.".format(
              dataset))
    dataset._input_datasets = [self]  # pylint: disable=protected-access
    return dataset

  def window(self, size, shift=None, stride=1, drop_remainder=False):
    """Returns a dataset of "windows".

    Each "window" is a dataset that contains a subset of elements of the
    input dataset. These are finite datasets of size `size` (or possibly fewer
    if there are not enough input elements to fill the window and
    `drop_remainder` evaluates to `False`).

    For example:

    >>> dataset = tf.data.Dataset.range(7).window(3)
    >>> for window in dataset:
    ...   print(window)
    <...Dataset shapes: (), types: tf.int64>
    <...Dataset shapes: (), types: tf.int64>
    <...Dataset shapes: (), types: tf.int64>

    Since windows are datasets, they can be iterated over:

    >>> for window in dataset:
    ...   print([item.numpy() for item in window])
    [0, 1, 2]
    [3, 4, 5]
    [6]

    #### Shift

    The `shift` argument determines the number of input elements to shift
    between the start of each window. If windows and elements are both numbered
    starting at 0, the first element in window `k` will be element `k * shift`
    of the input dataset. In particular, the first element of the first window
    will always be the first element of the input dataset.

    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1,
    ...                                           drop_remainder=True)
    >>> for window in dataset:
    ...   print(list(window.as_numpy_iterator()))
    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]
    [4, 5, 6]

    #### Stride

    The `stride` argument determines the stride between input elements within a
    window.

    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1, stride=2,
    ...                                           drop_remainder=True)
    >>> for window in dataset:
    ...   print(list(window.as_numpy_iterator()))
    [0, 2, 4]
    [1, 3, 5]
    [2, 4, 6]

    #### Nested elements

    When the `window` transformation is applied to a dataset whos elements are
    nested structures, it produces a dataset where the elements have the same
    nested structure but each leaf is replaced by a window. In other words,
    the nesting is applied outside of the windows as opposed inside of them.

    The type signature is:

    ```
    def window(
        self: Dataset[Nest[T]], ...
    ) -> Dataset[Nest[Dataset[T]]]
    ```

    Applying `window` to a `Dataset` of tuples gives a tuple of windows:

    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5],
    ...                                               [6, 7, 8, 9, 10]))
    >>> dataset = dataset.window(2)
    >>> windows = next(iter(dataset))
    >>> windows
    (<...Dataset shapes: (), types: tf.int32>,
     <...Dataset shapes: (), types: tf.int32>)

    >>> def to_numpy(ds):
    ...   return list(ds.as_numpy_iterator())
    >>>
    >>> for windows in dataset:
    ...   print(to_numpy(windows[0]), to_numpy(windows[1]))
    [1, 2] [6, 7]
    [3, 4] [8, 9]
    [5] [10]

    Applying `window` to a `Dataset` of dictionaries gives a dictionary of
    `Datasets`:

    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3],
    ...                                               'b': [4, 5, 6],
    ...                                               'c': [7, 8, 9]})
    >>> dataset = dataset.window(2)
    >>> def to_numpy(ds):
    ...   return list(ds.as_numpy_iterator())
    >>>
    >>> for windows in dataset:
    ...   print(tf.nest.map_structure(to_numpy, windows))
    {'a': [1, 2], 'b': [4, 5], 'c': [7, 8]}
    {'a': [3], 'b': [6], 'c': [9]}

    #### Flatten a dataset of windows

    The `Dataset.flat_map` and `Dataset.interleave` methods can be used to
    flatten a dataset of windows into a single dataset.

    The argument to `flat_map` is a function that takes an element from the
    dataset and returns a `Dataset`. `flat_map` chains together the resulting
    datasets sequentially.

    For example, to turn each window into a dense tensor:

    >>> size = 3
    >>> dataset = tf.data.Dataset.range(7).window(size, shift=1,
    ...                                           drop_remainder=True)
    >>> batched = dataset.flat_map(lambda x:x.batch(3))
    >>> for batch in batched:
    ...   print(batch.numpy())
    [0 1 2]
    [1 2 3]
    [2 3 4]
    [3 4 5]
    [4 5 6]

    Args:
      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
        of the input dataset to combine into a window. Must be positive.
      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of input elements by which the window moves in each iteration.
        Defaults to `size`. Must be positive.
      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        stride of the input elements in the sliding window. Must be positive.
        The default value of 1 means "retain every input element".
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last windows should be dropped if their size is smaller than
        `size`.

    Returns:
      Dataset: A `Dataset` of (nests of) windows. Each window is a finite
        datasets of flat elements.
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

    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1).numpy()
    5
    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y).numpy()
    10

    Args:
      initial_state: An element representing the initial state of the
        transformation.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. It must take two arguments and return a new element
        The structure of `new_state` must match the structure of
        `initial_state`.

    Returns:
      A dataset element corresponding to the final state of the transformation.

    """

    with ops.name_scope("initial_state"):
      initial_state = structure.normalize_element(initial_state)
    state_structure = structure.type_spec_from_value(initial_state)

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `state_structure`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = StructuredFunctionWrapper(
          reduce_func,
          "reduce()",
          input_structure=(state_structure, self.element_spec),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      output_classes = wrapped_func.output_classes
      state_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
          state_structure)
      for new_state_class, state_class in zip(
          nest.flatten(output_classes), nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (state_classes, wrapped_func.output_classes))

      # Extract and validate type information from the returned values.
      output_types = wrapped_func.output_types
      state_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
          state_structure)
      for new_state_type, state_type in zip(
          nest.flatten(output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (state_types, wrapped_func.output_types))

      # Extract shape information from the returned values.
      output_shapes = wrapped_func.output_shapes
      state_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
          state_structure)
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
        # TODO(b/110122868): Support a "most specific compatible structure"
        # method for combining structures, to avoid using legacy structures
        # here.
        state_structure = structure.convert_legacy_structure(
            state_types,
            nest.pack_sequence_as(state_shapes, weakened_state_shapes),
            state_classes)

    reduce_func = wrapped_func.function
    reduce_func.add_to_graph(ops.get_default_graph())

    dataset = self._apply_debug_options()

    # pylint: disable=protected-access
    return structure.from_compatible_tensor_list(
        state_structure,
        gen_dataset_ops.reduce_dataset(
            dataset._variant_tensor,
            structure.to_tensor_list(state_structure, initial_state),
            reduce_func.captured_inputs,
            f=reduce_func,
            output_shapes=structure.get_flat_tensor_shapes(state_structure),
            output_types=structure.get_flat_tensor_types(state_structure)))

  def get_single_element(self):
    """Returns the single element of the `dataset` as a nested structure of tensors.

    The function enables you to use a `tf.data.Dataset` in a stateless
    "tensor-in tensor-out" expression, without creating an iterator.
    This facilitates the ease of data transformation on tensors using the
    optimized `tf.data.Dataset` abstraction on top of them.

    For example, lets consider a `preprocessing_fn` which would take as an
    input the raw features and returns the processed feature along with
    it's label.

    ```python
    def preprocessing_fn(raw_feature):
      # ... the raw_feature is preprocessed as per the use-case
      return feature

    raw_features = ...  # input batch of BATCH_SIZE elements.
    dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
              .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
              .batch(BATCH_SIZE))

    processed_features = dataset.get_single_element()
    ```

    In the above example, the `raw_features` tensor of length=BATCH_SIZE
    was converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was
    mapped using the `preprocessing_fn` and the processed features were
    grouped into a single batch. The final `dataset` contains only one element
    which is a batch of all the processed features.

    NOTE: The `dataset` should contain only one element.

    Now, instead of creating an iterator for the `dataset` and retrieving the
    batch of features, the `tf.data.get_single_element()` function is used
    to skip the iterator creation process and directly output the batch of
    features.

    This can be particularly useful when your tensor transformations are
    expressed as `tf.data.Dataset` operations, and you want to use those
    transformations while serving your model.

    #### Keras

    ```python

    model = ... # A pre-built or custom model

    class PreprocessingModel(tf.keras.Model):
      def __init__(self, model):
        super().__init__(self)
        self.model = model

      @tf.function(input_signature=[...])
      def serving_fn(self, data):
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
        ds = ds.batch(batch_size=BATCH_SIZE)
        return tf.argmax(self.model(ds.get_single_element()), axis=-1)

    preprocessing_model = PreprocessingModel(model)
    your_exported_model_dir = ... # save the model to this path.
    tf.saved_model.save(preprocessing_model, your_exported_model_dir,
                  signatures={'serving_default': preprocessing_model.serving_fn}
                  )
    ```

    #### Estimator

    In the case of estimators, you need to generally define a `serving_input_fn`
    which would require the features to be processed by the model while
    inferencing.

    ```python
    def serving_input_fn():

      raw_feature_spec = ... # Spec for the raw_features
      input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
          raw_feature_spec, default_batch_size=None)
      )
      serving_input_receiver = input_fn()
      raw_features = serving_input_receiver.features

      def preprocessing_fn(raw_feature):
        # ... the raw_feature is preprocessed as per the use-case
        return feature

      dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
                .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
                .batch(BATCH_SIZE))

      processed_features = dataset.get_single_element()

      # Please note that the value of `BATCH_SIZE` should be equal to
      # the size of the leading dimension of `raw_features`. This ensures
      # that `dataset` has only element, which is a pre-requisite for
      # using `dataset.get_single_element()`.

      return tf.estimator.export.ServingInputReceiver(
          processed_features, serving_input_receiver.receiver_tensors)

    estimator = ... # A pre-built or custom estimator
    estimator.export_saved_model(your_exported_model_dir, serving_input_fn)
    ```

    Returns:
      A nested structure of `tf.Tensor` objects, corresponding to the single
      element of `dataset`.

    Raises:
      InvalidArgumentError: (at runtime) if `dataset` does not contain exactly
        one element.
    """

    return structure.from_compatible_tensor_list(
        self.element_spec,
        gen_dataset_ops.dataset_to_single_element(self._variant_tensor,
                                                  **self._flat_structure))  # pylint: disable=protected-access

  def unbatch(self):
    """Splits elements of a dataset into multiple elements.

    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary for each input element, then for each element in the
    dataset, the unbatched dataset will contain `B` consecutive elements
    of shape `[a0, a1, ...]`.

    >>> elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]
    >>> dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)
    >>> dataset = dataset.unbatch()
    >>> list(dataset.as_numpy_iterator())
    [1, 2, 3, 1, 2, 1, 2, 3, 4]

    Note: `unbatch` requires a data copy to slice up the batched tensor into
    smaller, unbatched tensors. When optimizing performance, try to avoid
    unnecessary usage of `unbatch`.

    Returns:
      A `Dataset`.
    """
    normalized_dataset = normalize_to_dense(self)
    return _UnbatchDataset(normalized_dataset)

  def with_options(self, options):
    """Returns a new `tf.data.Dataset` with the given options set.

    The options are "global" in the sense they apply to the entire dataset.
    If options are set multiple times, they are merged as long as different
    options do not use different non-default values.

    >>> ds = tf.data.Dataset.range(5)
    >>> ds = ds.interleave(lambda x: tf.data.Dataset.range(5),
    ...                    cycle_length=3,
    ...                    num_parallel_calls=3)
    >>> options = tf.data.Options()
    >>> # This will make the interleave order non-deterministic.
    >>> options.experimental_deterministic = False
    >>> ds = ds.with_options(options)

    Args:
      options: A `tf.data.Options` that identifies the options the use.

    Returns:
      Dataset: A `Dataset` with the given options.

    Raises:
      ValueError: when an option is set more than once to a non-default value
    """
    return _OptionsDataset(self, options)

  def cardinality(self):
    """Returns the cardinality of the dataset, if known.

    `cardinality` may return `tf.data.INFINITE_CARDINALITY` if the dataset
    contains an infinite number of elements or `tf.data.UNKNOWN_CARDINALITY` if
    the analysis fails to determine the number of elements in the dataset
    (e.g. when the dataset source is a file).

    >>> dataset = tf.data.Dataset.range(42)
    >>> print(dataset.cardinality().numpy())
    42
    >>> dataset = dataset.repeat()
    >>> cardinality = dataset.cardinality()
    >>> print((cardinality == tf.data.INFINITE_CARDINALITY).numpy())
    True
    >>> dataset = dataset.filter(lambda x: True)
    >>> cardinality = dataset.cardinality()
    >>> print((cardinality == tf.data.UNKNOWN_CARDINALITY).numpy())
    True

    Returns:
      A scalar `tf.int64` `Tensor` representing the cardinality of the dataset.
      If the cardinality is infinite or unknown, `cardinality` returns the
      named constants `tf.data.INFINITE_CARDINALITY` and
      `tf.data.UNKNOWN_CARDINALITY` respectively.
    """
    return gen_dataset_ops.dataset_cardinality(self._variant_tensor)

  def group_by_window(self,
                      key_func,
                      reduce_func,
                      window_size=None,
                      window_size_func=None):
    """Groups windows of elements by key and reduces them.

    This transformation maps each consecutive element in a dataset to a key
    using `key_func` and groups the elements by key. It then applies
    `reduce_func` to at most `window_size_func(key)` elements matching the same
    key. All except the final window for each key will contain
    `window_size_func(key)` elements; the final window may be smaller.

    You may provide either a constant `window_size` or a window size determined
    by the key through `window_size_func`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> window_size = 5
    >>> key_func = lambda x: x%2
    >>> reduce_func = lambda key, dataset: dataset.batch(window_size)
    >>> dataset = dataset.group_by_window(
    ...           key_func=key_func,
    ...           reduce_func=reduce_func,
    ...           window_size=window_size)
    >>> for elem in dataset.as_numpy_iterator():
    ...   print(elem)
    [0 2 4 6 8]
    [1 3 5 7 9]

    Args:
      key_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        scalar `tf.int64` tensor.
      reduce_func: A function mapping a key and a dataset of up to `window_size`
        consecutive elements matching that key to another dataset.
      window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements matching the same key to combine in a single batch,
        which will be passed to `reduce_func`. Mutually exclusive with
        `window_size_func`.
      window_size_func: A function mapping a key to a `tf.int64` scalar
        `tf.Tensor`, representing the number of consecutive elements matching
        the same key to combine in a single batch, which will be passed to
        `reduce_func`. Mutually exclusive with `window_size`.

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if neither or both of {`window_size`, `window_size_func`} are
        passed.
    """
    if (window_size is not None and window_size_func or
        not (window_size is not None or window_size_func)):
      raise ValueError("Must pass either window_size or window_size_func.")

    if window_size is not None:

      def constant_window_func(unused_key):
        return ops.convert_to_tensor(window_size, dtype=dtypes.int64)

      window_size_func = constant_window_func

    assert window_size_func is not None

    return _GroupByWindowDataset(self, key_func, reduce_func, window_size_func)

  def bucket_by_sequence_length(self,
                                element_length_func,
                                bucket_boundaries,
                                bucket_batch_sizes,
                                padded_shapes=None,
                                padding_values=None,
                                pad_to_bucket_boundary=False,
                                no_padding=False,
                                drop_remainder=False):
    """A transformation that buckets elements in a `Dataset` by length.

    Elements of the `Dataset` are grouped together by length and then are padded
    and batched.

    This is useful for sequence tasks in which the elements have variable
    length. Grouping together elements that have similar lengths reduces the
    total fraction of padding in a batch which increases training step
    efficiency.

    Below is an example to bucketize the input data to the 3 buckets
    "[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.

    >>> elements = [
    ...   [0], [1, 2, 3, 4], [5, 6, 7],
    ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, tf.int64, output_shapes=[None])
    >>> dataset = dataset.bucket_by_sequence_length(
    ...         element_length_func=lambda elem: tf.shape(elem)[0],
    ...         bucket_boundaries=[3, 5],
    ...         bucket_batch_sizes=[2, 2, 2])
    >>> for elem in dataset.as_numpy_iterator():
    ...   print(elem)
    [[1 2 3 4]
    [5 6 7 0]]
    [[ 7  8  9 10 11  0]
    [13 14 15 16 19 20]]
    [[ 0  0]
    [21 22]]

    Args:
      element_length_func: function from element in `Dataset` to `tf.int32`,
        determines the length of the element, which will determine the bucket it
        goes into.
      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
        `len(bucket_boundaries) + 1`.
      padded_shapes: Nested structure of `tf.TensorShape` to pass to
        `tf.data.Dataset.padded_batch`. If not provided, will use
        `dataset.output_shapes`, which will result in variable length dimensions
        being padded out to the maximum length in each batch.
      padding_values: Values to pad with, passed to
        `tf.data.Dataset.padded_batch`. Defaults to padding with 0.
      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
        size to maximum length in batch. If `True`, will pad dimensions with
        unknown size to bucket boundary minus 1 (i.e., the maximum length in
        each bucket), and caller must ensure that the source `Dataset` does not
        contain any elements with length longer than `max(bucket_boundaries)`.
      no_padding: `bool`, indicates whether to pad the batch features (features
        need to be either of type `tf.sparse.SparseTensor` or of same shape).
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
    """
    if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
      raise ValueError(
          "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1")

    batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

    def element_to_bucket_id(*args):
      """Return int64 id of the length bucket for this element."""
      seq_length = element_length_func(*args)

      boundaries = list(bucket_boundaries)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = math_ops.logical_and(
          math_ops.less_equal(buckets_min, seq_length),
          math_ops.less(seq_length, buckets_max))
      bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))

      return bucket_id

    def window_size_fn(bucket_id):
      # The window size is set to the batch size for this bucket
      window_size = batch_sizes[bucket_id]
      return window_size

    def make_padded_shapes(shapes, none_filler=None):
      padded = []
      for shape in nest.flatten(shapes):
        shape = tensor_shape.TensorShape(shape)
        shape = [
            none_filler if tensor_shape.dimension_value(d) is None else d
            for d in shape
        ]
        padded.append(shape)
      return nest.pack_sequence_as(shapes, padded)

    def batching_fn(bucket_id, grouped_dataset):
      """Batch elements in dataset."""
      batch_size = window_size_fn(bucket_id)
      if no_padding:
        return grouped_dataset.batch(batch_size, drop_remainder=drop_remainder)
      none_filler = None
      if pad_to_bucket_boundary:
        err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                   "length < max(bucket_boundaries).")
        check = check_ops.assert_less(
            bucket_id,
            constant_op.constant(
                len(bucket_batch_sizes) - 1, dtype=dtypes.int64),
            message=err_msg)
        with ops.control_dependencies([check]):
          boundaries = constant_op.constant(
              bucket_boundaries, dtype=dtypes.int64)
          bucket_boundary = boundaries[bucket_id]
          none_filler = bucket_boundary - 1
      input_shapes = get_legacy_output_shapes(grouped_dataset)
      shapes = make_padded_shapes(
          padded_shapes or input_shapes, none_filler=none_filler)
      return grouped_dataset.padded_batch(
          batch_size, shapes, padding_values, drop_remainder=drop_remainder)

    return self.group_by_window(
        key_func=element_to_bucket_id,
        reduce_func=batching_fn,
        window_size_func=window_size_fn)

  @staticmethod
  def random(seed=None):
    """Creates a `Dataset` of pseudorandom values.

    The dataset generates a sequence of uniformly distributed integer values.

    >>> ds1 = tf.data.Dataset.random(seed=4).take(10)
    >>> ds2 = tf.data.Dataset.random(seed=4).take(10)
    >>> print(list(ds2.as_numpy_iterator())==list(ds2.as_numpy_iterator()))
    True

    Args:
      seed: (Optional) If specified, the dataset produces a deterministic
        sequence of values.

    Returns:
      Dataset: A `Dataset`.
    """
    return RandomDataset(seed=seed)

  def snapshot(self,
               path,
               compression="AUTO",
               reader_func=None,
               shard_func=None):
    """API to persist the output of the input dataset.

    The snapshot API allows users to transparently persist the output of their
    preprocessing pipeline to disk, and materialize the pre-processed data on a
    different training run.

    This API enables repeated preprocessing steps to be consolidated, and allows
    re-use of already processed data, trading off disk storage and network
    bandwidth for freeing up more valuable CPU resources and accelerator compute
    time.

    https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md
    has detailed design documentation of this feature.

    Users can specify various options to control the behavior of snapshot,
    including how snapshots are read from and written to by passing in
    user-defined functions to the `reader_func` and `shard_func` parameters.

    `shard_func` is a user specified function that maps input elements to
    snapshot shards.

    Users may want to specify this function to control how snapshot files should
    be written to disk. Below is an example of how a potential `shard_func`
    could be written.

    ```python
    dataset = ...
    dataset = dataset.enumerate()
    dataset = dataset.snapshot("/path/to/snapshot/dir",
        shard_func=lambda x, y: x % NUM_SHARDS, ...)
    dataset = dataset.map(lambda x, y: y)
    ```

    `reader_func` is a user specified function that accepts a single argument:
    (1) a Dataset of Datasets, each representing a "split" of elements of the
    original dataset. The cardinality of the input dataset matches the
    number of the shards specified in the `shard_func` (see above). The function
    should return a Dataset of elements of the original dataset.

    Users may want specify this function to control how snapshot files should be
    read from disk, including the amount of shuffling and parallelism.

    Here is an example of a standard reader function a user can define. This
    function enables both dataset shuffling and parallel reading of datasets:

    ```python
    def user_reader_func(datasets):
      # shuffle the datasets splits
      datasets = datasets.shuffle(NUM_CORES)
      # read datasets in parallel and interleave their elements
      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

    dataset = dataset.snapshot("/path/to/snapshot/dir",
        reader_func=user_reader_func)
    ```

    By default, snapshot parallelizes reads by the number of cores available on
    the system, but will not attempt to shuffle the data.

    Args:
      path: Required. A directory to use for storing / loading the snapshot to /
        from.
      compression: Optional. The type of compression to apply to the snapshot
        written to disk. Supported options are `GZIP`, `SNAPPY`, `AUTO` or None.
        Defaults to `AUTO`, which attempts to pick an appropriate compression
        algorithm for the dataset.
      reader_func: Optional. A function to control how to read data from
        snapshot shards.
      shard_func: Optional. A function to control how to shard data when writing
        a snapshot.

    Returns:
      A `Dataset`.
    """

    project_func = None
    input_dataset = self
    if shard_func is None:
      input_dataset = input_dataset.enumerate()
      # This sets the amount of parallelism based on the number of CPU cores on
      # the machine where this Python code is executed, which may differ from
      # the number of CPU cores where the input pipeline graph is actually
      # executed (e.g. remote Cloud TPU workers).
      local_shard_func = lambda index, _: index % multiprocessing.cpu_count()
      project_func = lambda _, elem: elem
    else:
      local_shard_func = shard_func
    dataset = _SnapshotDataset(
        input_dataset=input_dataset,
        path=path,
        compression=compression,
        reader_func=reader_func,
        # This will not do the right thing where the graph is built on a
        # different machine than the executor (e.g. Cloud TPUs).
        shard_func=local_shard_func)
    if project_func is not None:
      dataset = dataset.map(project_func)
    return dataset

  def scan(self, initial_state, scan_func):
    """A transformation that scans a function across an input dataset.

    This transformation is a stateful relative of `tf.data.Dataset.map`.
    In addition to mapping `scan_func` across the elements of the input dataset,
    `scan()` accumulates one or more state tensors, whose initial values are
    `initial_state`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> initial_state = tf.constant(0, dtype=tf.int64)
    >>> scan_func = lambda state, i: (state + i, state + i)
    >>> dataset = dataset.scan(initial_state=initial_state, scan_func=scan_func)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

    Args:
      initial_state: A nested structure of tensors, representing the initial
        state of the accumulator.
      scan_func: A function that maps `(old_state, input_element)` to
        `(new_state, output_element)`. It must take two arguments and return a
        pair of nested structures of tensors. The `new_state` must match the
        structure of `initial_state`.

    Returns:
      A `Dataset`.
    """

    return _ScanDataset(self, initial_state=initial_state, scan_func=scan_func)

  def take_while(self, predicate):
    """A transformation that stops dataset iteration based on a `predicate`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.take_while(lambda x: x < 5)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2, 3, 4]

    Args:
      predicate: A function that maps a nested structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.bool` tensor.

    Returns:
      A `Dataset`.
    """

    return _TakeWhileDataset(self, predicate)

  def unique(self):
    """A transformation that discards duplicate elements of a `Dataset`.

    Use this transformation to produce a dataset that contains one instance of
    each unique element in the input. For example:

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])
    >>> dataset = dataset.unique()
    >>> sorted(list(dataset.as_numpy_iterator()))
    [1, 2, 37]

    Note: This transformation only supports datasets which fit into memory
    and have elements of either `tf.int32`, `tf.int64` or `tf.string` type.

    Returns:
      A `Dataset`.
    """

    return _UniqueDataset(self)


@tf_export(v1=["data.Dataset"])
class DatasetV1(DatasetV2):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

  def __init__(self):
    try:
      variant_tensor = self._as_variant_tensor()
    except AttributeError as e:
      if "_as_variant_tensor" in str(e):
        raise AttributeError("Please use _variant_tensor instead of "
                             "_as_variant_tensor() to obtain the variant "
                             "associated with a dataset")
      raise AttributeError("{}: A likely cause of this error is that the super "
                           "call for this dataset is not the last line of the "
                           "__init__ method. The base class causes the "
                           "_as_variant_tensor call in its constructor and "
                           "if that uses attributes defined in the __init__ "
                           "method, those attrs need to be defined before the "
                           "super call.".format(e))
    super(DatasetV1, self).__init__(variant_tensor)

  @abc.abstractmethod
  def _as_variant_tensor(self):
    """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
    raise NotImplementedError("Dataset._as_variant_tensor")

  @deprecation.deprecated(
      None, "This is a deprecated API that should only be used in TF 1 graph "
      "mode and legacy TF 2 graph mode available through `tf.compat.v1`. In "
      "all other situations -- namely, eager mode and inside `tf.function` -- "
      "you can consume dataset elements using `for elem in dataset: ...` or "
      "by explicitly creating iterator via `iterator = iter(dataset)` and "
      "fetching its elements via `values = next(iterator)`. Furthermore, "
      "this API is not available in TF 2. During the transition from TF 1 "
      "to TF 2 you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)` "
      "to create a TF 1 graph mode style iterator for a dataset created "
      "through TF 2 APIs. Note that this should be a transient state of your "
      "code base as there are in general no guarantees about the "
      "interoperability of TF 1 and TF 2 code.")
  def make_one_shot_iterator(self):
    """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization. For
    that see `make_initializable_iterator`.

    Example:

    ```python
    # Building graph ...
    dataset = ...
    next_value = dataset.make_one_shot_iterator().get_next()

    # ... from within a session ...
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Returns:
      An `tf.data.Iterator` for elements of this dataset.
    """
    return self._make_one_shot_iterator()

  def _make_one_shot_iterator(self):  # pylint: disable=missing-docstring
    if context.executing_eagerly():
      with ops.colocate_with(self._variant_tensor):
        return iterator_ops.OwnedIterator(self)

    _ensure_same_dataset_graph(self)
    # Some ops (e.g. dataset ops) are marked as stateful but are stil safe to
    # to capture by value. We must allowlist these ops so that the capturing
    # logic captures the ops instead of raising an exception.
    allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
    graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(
        capture_by_value=True,
        allowlisted_stateful_ops=allowlisted_stateful_ops)
    def _make_dataset():
      """Factory function for a dataset."""
      # NOTE(mrry): `Defun` does not capture the graph-level seed from the
      # enclosing graph, so if a graph-level seed is present we set the local
      # graph seed based on a combination of the graph- and op-level seeds.
      if graph_level_seed is not None:
        assert op_level_seed is not None
        core_random_seed.set_random_seed(
            (graph_level_seed + 87654321 * op_level_seed) % (2 ** 63 - 1))

      dataset = self._apply_debug_options()
      return dataset._variant_tensor  # pylint: disable=protected-access

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

    with ops.colocate_with(self._variant_tensor):
      # pylint: disable=protected-access
      return iterator_ops.Iterator(
          gen_dataset_ops.one_shot_iterator(
              dataset_factory=_make_dataset, **self._flat_structure), None,
          get_legacy_output_types(self), get_legacy_output_shapes(self),
          get_legacy_output_classes(self))

  @deprecation.deprecated(
      None, "This is a deprecated API that should only be used in TF 1 graph "
      "mode and legacy TF 2 graph mode available through `tf.compat.v1`. "
      "In all other situations -- namely, eager mode and inside `tf.function` "
      "-- you can consume dataset elements using `for elem in dataset: ...` "
      "or by explicitly creating iterator via `iterator = iter(dataset)` "
      "and fetching its elements via `values = next(iterator)`. "
      "Furthermore, this API is not available in TF 2. During the transition "
      "from TF 1 to TF 2 you can use "
      "`tf.compat.v1.data.make_initializable_iterator(dataset)` to create a TF "
      "1 graph mode style iterator for a dataset created through TF 2 APIs. "
      "Note that this should be a transient state of your code base as there "
      "are in general no guarantees about the interoperability of TF 1 and TF "
      "2 code.")
  def make_initializable_iterator(self, shared_name=None):
    """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    # Building graph ...
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    next_value = iterator.get_next()  # This is a Tensor.

    # ... from within a session ...
    sess.run(iterator.initializer)
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Args:
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).

    Returns:
      A `tf.data.Iterator` for elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    return self._make_initializable_iterator(shared_name)

  def _make_initializable_iterator(self, shared_name=None):  # pylint: disable=missing-docstring
    if context.executing_eagerly():
      raise RuntimeError(
          "dataset.make_initializable_iterator is not supported when eager "
          "execution is enabled. Use `for element in dataset` instead.")
    _ensure_same_dataset_graph(self)
    dataset = self._apply_debug_options()
    if shared_name is None:
      shared_name = ""

    with ops.colocate_with(self._variant_tensor):
      iterator_resource = gen_dataset_ops.iterator_v2(
          container="", shared_name=shared_name, **self._flat_structure)

      initializer = gen_dataset_ops.make_iterator(
          dataset._variant_tensor,  # pylint: disable=protected-access
          iterator_resource)

      # pylint: disable=protected-access
      return iterator_ops.Iterator(iterator_resource, initializer,
                                   get_legacy_output_types(dataset),
                                   get_legacy_output_shapes(dataset),
                                   get_legacy_output_classes(dataset))

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(dataset)`.")
  def output_classes(self):
    """Returns the class of each component of an element of this dataset.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(dataset)`.")
  def output_shapes(self):
    """Returns the shape of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(dataset)`.")
  def output_types(self):
    """Returns the type of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  def element_spec(self):
    # TODO(b/110122868): Remove this override once all `Dataset` instances
    # implement `element_structure`.
    return structure.convert_legacy_structure(
        self.output_types, self.output_shapes, self.output_classes)

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
    """Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.sparse.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
    return DatasetV1Adapter(SparseTensorSliceDataset(sparse_tensor))

  @staticmethod
  @functools.wraps(DatasetV2.from_generator)
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None):
    # Calling DatasetV2.from_generator with output_shapes or output_types is
    # deprecated, but this is already checked by the decorator on this function.
    with deprecation.silence():
      return DatasetV1Adapter(
          DatasetV2.from_generator(generator, output_types, output_shapes, args,
                                   output_signature))

  @staticmethod
  @functools.wraps(DatasetV2.range)
  def range(*args, **kwargs):
    return DatasetV1Adapter(DatasetV2.range(*args, **kwargs))

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

  @functools.wraps(DatasetV2.shard)
  def shard(self, num_shards, index):
    return DatasetV1Adapter(super(DatasetV1, self).shard(num_shards, index))

  @functools.wraps(DatasetV2.batch)
  def batch(self,
            batch_size,
            drop_remainder=False,
            num_parallel_calls=None,
            deterministic=None):
    return DatasetV1Adapter(
        super(DatasetV1, self).batch(batch_size, drop_remainder,
                                     num_parallel_calls, deterministic))

  @functools.wraps(DatasetV2.padded_batch)
  def padded_batch(self,
                   batch_size,
                   padded_shapes=None,
                   padding_values=None,
                   drop_remainder=False):
    return DatasetV1Adapter(
        super(DatasetV1, self).padded_batch(batch_size, padded_shapes,
                                            padding_values, drop_remainder))

  @functools.wraps(DatasetV2.map)
  def map(self, map_func, num_parallel_calls=None, deterministic=None):
    if num_parallel_calls is None or DEBUG_MODE:
      return DatasetV1Adapter(
          MapDataset(self, map_func, preserve_cardinality=False))
    else:
      return DatasetV1Adapter(
          ParallelMapDataset(
              self,
              map_func,
              num_parallel_calls,
              deterministic,
              preserve_cardinality=False))

  @deprecation.deprecated(None, "Use `tf.data.Dataset.map()")
  def map_with_legacy_function(self,
                               map_func,
                               num_parallel_calls=None,
                               deterministic=None):
    """Maps `map_func` across the elements of this dataset.

    Note: This is an escape hatch for existing uses of `map` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `map` as this method will be removed in V2.

    Args:
      map_func: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to another (nested) structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, this
        boolean controls the order in which the transformation produces
        elements. If set to `False`, the transformation is allowed to yield
        elements out of order to trade determinism for performance. If not
        specified, the `tf.data.Options.experimental_deterministic` option
        (`True` by default) controls the behavior.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None:
      if deterministic is not None:
        warnings.warn("The `deterministic` argument has no effect unless the "
                      "`num_parallel_calls` argument is specified.")
      return DatasetV1Adapter(
          MapDataset(
              self,
              map_func,
              preserve_cardinality=False,
              use_legacy_function=True))
    else:
      return DatasetV1Adapter(
          ParallelMapDataset(
              self,
              map_func,
              num_parallel_calls,
              deterministic,
              preserve_cardinality=False,
              use_legacy_function=True))

  @functools.wraps(DatasetV2.flat_map)
  def flat_map(self, map_func):
    return DatasetV1Adapter(super(DatasetV1, self).flat_map(map_func))

  @functools.wraps(DatasetV2.interleave)
  def interleave(self,
                 map_func,
                 cycle_length=None,
                 block_length=None,
                 num_parallel_calls=None,
                 deterministic=None):
    return DatasetV1Adapter(
        super(DatasetV1, self).interleave(map_func, cycle_length, block_length,
                                          num_parallel_calls, deterministic))

  @functools.wraps(DatasetV2.filter)
  def filter(self, predicate):
    return DatasetV1Adapter(super(DatasetV1, self).filter(predicate))

  @deprecation.deprecated(None, "Use `tf.data.Dataset.filter()")
  def filter_with_legacy_function(self, predicate):
    """Filters this dataset according to `predicate`.

    Note: This is an escape hatch for existing uses of `filter` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `filter` as this method will be removed in V2.

    Args:
      predicate: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.bool` tensor.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    """
    return FilterDataset(self, predicate, use_legacy_function=True)

  @functools.wraps(DatasetV2.apply)
  def apply(self, transformation_func):
    return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

  @functools.wraps(DatasetV2.window)
  def window(self, size, shift=None, stride=1, drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).window(
        size, shift, stride, drop_remainder))

  @functools.wraps(DatasetV2.unbatch)
  def unbatch(self):
    return DatasetV1Adapter(super(DatasetV1, self).unbatch())

  @functools.wraps(DatasetV2.with_options)
  def with_options(self, options):
    return DatasetV1Adapter(super(DatasetV1, self).with_options(options))


if tf2.enabled():
  Dataset = DatasetV2
else:
  Dataset = DatasetV1


class DatasetV1Adapter(DatasetV1):
  """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

  def __init__(self, dataset):
    self._dataset = dataset
    super(DatasetV1Adapter, self).__init__()

  def _as_variant_tensor(self):
    return self._dataset._variant_tensor  # pylint: disable=protected-access

  def _inputs(self):
    return self._dataset._inputs()  # pylint: disable=protected-access

  def _functions(self):
    return self._dataset._functions()  # pylint: disable=protected-access

  def options(self):
    return self._dataset.options()

  @property
  def element_spec(self):
    return self._dataset.element_spec  # pylint: disable=protected-access

  def __iter__(self):
    return iter(self._dataset)


def _ensure_same_dataset_graph(dataset):
  """Walks the dataset graph to ensure all datasets come from the same graph."""
  # pylint: disable=protected-access
  current_graph = ops.get_default_graph()
  bfs_q = Queue.Queue()
  bfs_q.put(dataset)
  visited = []
  while not bfs_q.empty():
    ds = bfs_q.get()
    visited.append(ds)
    ds_graph = ds._graph
    if current_graph != ds_graph:
      raise ValueError(
          "The graph (" + str(current_graph) + ") of the iterator is different "
          "from the graph (" + str(ds_graph) + ") the dataset: " +
          str(ds._variant_tensor) + " was  created in. If you are using the "
          "Estimator API, make sure that no part of the dataset returned by "
          "the `input_fn` function is defined outside the `input_fn` function. "
          "Please ensure that all datasets in the pipeline are created in the "
          "same graph as the iterator.")
    for input_ds in ds._inputs():
      if input_ds not in visited:
        bfs_q.put(input_ds)


@tf_export(v1=["data.make_one_shot_iterator"])
def make_one_shot_iterator(dataset):
  """Creates an iterator for elements of `dataset`.

  Note: The returned iterator will be initialized automatically.
  A "one-shot" iterator does not support re-initialization.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A `tf.data.Iterator` for elements of `dataset`.

  @compatibility(TF2)
  This is a legacy API for consuming dataset elements and should only be used
  during transition from TF 1 to TF 2. Note that using this API should be
  a transient state of your code base as there are in general no guarantees
  about the interoperability of TF 1 and TF 2 code.

  In TF 2 datasets are Python iterables which means you can consume their
  elements using `for elem in dataset: ...` or by explicitly creating iterator
  via `iterator = iter(dataset)` and fetching its elements via
  `values = next(iterator)`.
  @end_compatibility
  """
  try:
    # Call the defined `_make_one_shot_iterator()` if there is one, because some
    # datasets (e.g. for prefetching) override its behavior.
    return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_one_shot_iterator()  # pylint: disable=protected-access


@tf_export(v1=["data.make_initializable_iterator"])
def make_initializable_iterator(dataset, shared_name=None):
  """Creates an iterator for elements of `dataset`.

  Note: The returned iterator will be in an uninitialized state,
  and you must run the `iterator.initializer` operation before using it:

  ```python
  dataset = ...
  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  # ...
  sess.run(iterator.initializer)
  ```

  Args:
    dataset: A `tf.data.Dataset`.
    shared_name: (Optional.) If non-empty, the returned iterator will be shared
      under the given name across multiple sessions that share the same devices
      (e.g. when using a remote server).

  Returns:
    A `tf.data.Iterator` for elements of `dataset`.

  Raises:
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  This is a legacy API for consuming dataset elements and should only be used
  during transition from TF 1 to TF 2. Note that using this API should be
  a transient state of your code base as there are in general no guarantees
  about the interoperability of TF 1 and TF 2 code.

  In TF 2 datasets are Python iterables which means you can consume their
  elements using `for elem in dataset: ...` or by explicitly creating iterator
  via `iterator = iter(dataset)` and fetching its elements via
  `values = next(iterator)`.
  @end_compatibility
  """
  try:
    # Call the defined `_make_initializable_iterator()` if there is one, because
    # some datasets (e.g. for prefetching) override its behavior.
    return dataset._make_initializable_iterator(shared_name)  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_initializable_iterator(shared_name)  # pylint: disable=protected-access


@tf_export("data.experimental.get_structure")
def get_structure(dataset_or_iterator):
  """Returns the type signature for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or an `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.TypeSpec` objects matching the structure of an
    element of `dataset_or_iterator` and specifying the type of individual
    components.

  Raises:
    TypeError: If input is not a `tf.data.Dataset` or an `tf.data.Iterator`
      object.
  """
  try:
    return dataset_or_iterator.element_spec  # pylint: disable=protected-access
  except AttributeError:
    raise TypeError("`dataset_or_iterator` must be a `tf.data.Dataset` or "
                    "tf.data.Iterator object, but got %s." %
                    type(dataset_or_iterator))


@tf_export(v1=["data.get_output_classes"])
def get_legacy_output_classes(dataset_or_iterator):
  """Returns the output classes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of Python `type` objects matching the structure of the
    dataset / iterator elements and specifying the class of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_shapes"])
def get_legacy_output_shapes(dataset_or_iterator):
  """Returns the output shapes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.TensorShape` objects matching the structure of
    the dataset / iterator elements and specifying the shape of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_types"])
def get_legacy_output_types(dataset_or_iterator):
  """Returns the output shapes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.DType` objects matching the structure of
    dataset / iterator elements and specifying the shape of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export("data.Options")
class Options(options_lib.OptionsBase):
  """Represents options for `tf.data.Dataset`.

  A `tf.data.Options` object can be, for instance, used to control which static
  optimizations to apply to the input pipeline graph or whether to use
  performance modeling to dynamically tune the parallelism of operations such as
  `tf.data.Dataset.map` or `tf.data.Dataset.interleave`.

  The options are set for the entire dataset and are carried over to datasets
  created through tf.data transformations.

  The options can be set by constructing an `Options` object and using the
  `tf.data.Dataset.with_options(options)` transformation, which returns a
  dataset with the options set.

  >>> dataset = tf.data.Dataset.range(42)
  >>> options = tf.data.Options()
  >>> options.experimental_deterministic = False
  >>> dataset = dataset.with_options(options)
  >>> print(dataset.options().experimental_deterministic)
  False

  Note: A known limitation of the `tf.data.Options` implementation is that the
  options are not preserved across tf.function boundaries. In particular, to
  set options for a dataset that is iterated within a tf.function, the options
  need to be set within the same tf.function.
  """

  experimental_deterministic = options_lib.create_option(
      name="experimental_deterministic",
      ty=bool,
      docstring=
      "Whether the outputs need to be produced in deterministic order. If None,"
      " defaults to True.")

  experimental_distribute = options_lib.create_option(
      name="experimental_distribute",
      ty=distribute_options.DistributeOptions,
      docstring=
      "The distribution strategy options associated with the dataset. See "
      "`tf.data.experimental.DistributeOptions` for more details.",
      default_factory=distribute_options.DistributeOptions)

  experimental_optimization = options_lib.create_option(
      name="experimental_optimization",
      ty=optimization_options.OptimizationOptions,
      docstring=
      "The optimization options associated with the dataset. See "
      "`tf.data.experimental.OptimizationOptions` for more details.",
      default_factory=optimization_options.OptimizationOptions)

  experimental_slack = options_lib.create_option(
      name="experimental_slack",
      ty=bool,
      docstring="Whether to introduce 'slack' in the last `prefetch` of the "
      "input pipeline, if it exists. This may reduce CPU contention with "
      "accelerator host-side activity at the start of a step. The slack "
      "frequency is determined by the number of devices attached to this "
      "input pipeline. If None, defaults to False.")

  experimental_external_state_policy = options_lib.create_option(
      name="experimental_external_state_policy",
      ty=distribute_options.ExternalStatePolicy,
      docstring="This option can be used to override the default policy for "
      "how to handle external state when serializing a dataset or "
      "checkpointing its iterator. There are three settings available - "
      "IGNORE: External state is ignored without a warning; WARN: External "
      "state is ignored and a warning is logged; FAIL: External state results "
      "in an error.")

  threading = options_lib.create_option(
      name="threading",
      ty=threading_options.ThreadingOptions,
      docstring="The threading options associated with the dataset. See "
      "`tf.data.ThreadingOptions` for more details.",
      default_factory=threading_options.ThreadingOptions)

  def __getattr__(self, name):
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      return getattr(self, "threading")
    else:
      raise AttributeError("Attribute %s not found." % name)

  def __setattr__(self, name, value):
    if name == "experimental_threading":
      logging.warning("options.experimental_threading is deprecated. "
                      "Use options.threading instead.")
      super(Options, self).__setattr__("threading", value)
    else:
      super(Options, self).__setattr__(name, value)

  def _to_proto(self):
    pb = dataset_options_pb2.Options()
    if self.experimental_deterministic is not None:
      pb.deterministic = self.experimental_deterministic
    pb.distribute_options.CopyFrom(self.experimental_distribute._to_proto())  # pylint: disable=protected-access
    if self.experimental_external_state_policy is not None:
      pb.external_state_policy = (
          distribute_options.ExternalStatePolicy._to_proto(  # pylint: disable=protected-access
              self.experimental_external_state_policy))
    pb.optimization_options.CopyFrom(self.experimental_optimization._to_proto())  # pylint: disable=protected-access
    if self.experimental_slack is not None:
      pb.slack = self.experimental_slack
    pb.threading_options.CopyFrom(self.threading._to_proto())  # pylint: disable=protected-access
    return pb

  def _from_proto(self, pb):
    if pb.WhichOneof("optional_deterministic") is not None:
      self.experimental_deterministic = pb.deterministic
    self.experimental_distribute._from_proto(pb.distribute_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_external_state_policy") is not None:
      self.experimental_external_state_policy = (
          distribute_options.ExternalStatePolicy._from_proto(  # pylint: disable=protected-access
              pb.external_state_policy))
    self.experimental_optimization._from_proto(pb.optimization_options)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_slack") is not None:
      self.experimental_slack = pb.slack
    self.threading._from_proto(pb.threading_options)  # pylint: disable=protected-access

  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
    self.experimental_distribute._set_mutable(mutable)
    self.experimental_optimization._set_mutable(mutable)
    self.threading._set_mutable(mutable)

  def merge(self, options):
    """Merges itself with the given `tf.data.Options`.

    If this object and the `options` to merge set an option differently, a
    warning is generated and this object's value is updated with the `options`
    object's value.

    Args:
      options: a `tf.data.Options` to merge with

    Returns:
      New `tf.data.Options` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)


class DatasetSource(DatasetV2):
  """Abstract class representing a dataset with no inputs."""

  def _inputs(self):
    return []


class UnaryDataset(DatasetV2):
  """Abstract class representing a dataset with one input."""

  def __init__(self, input_dataset, variant_tensor):
    self._input_dataset = input_dataset
    super(UnaryDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]


class UnaryUnchangedStructureDataset(UnaryDataset):
  """Represents a unary dataset with the same input and output structure."""

  def __init__(self, input_dataset, variant_tensor):
    self._input_dataset = input_dataset
    super(UnaryUnchangedStructureDataset, self).__init__(
        input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._input_dataset.element_spec


class TensorDataset(DatasetSource):
  """A `Dataset` with a single element."""

  def __init__(self, element):
    """See `Dataset.from_tensors()` for details."""
    element = structure.normalize_element(element)
    self._structure = structure.type_spec_from_value(element)
    self._tensors = structure.to_tensor_list(self._structure, element)

    variant_tensor = gen_dataset_ops.tensor_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(TensorDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class TensorSliceDataset(DatasetSource):
  """A `Dataset` of slices from a dataset element."""

  def __init__(self, element):
    """See `Dataset.from_tensor_slices()` for details."""
    element = structure.normalize_element(element)
    batched_spec = structure.type_spec_from_value(element)
    self._tensors = structure.to_batched_tensor_list(batched_spec, element)
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(), batched_spec)  # pylint: disable=protected-access

    batch_dim = tensor_shape.Dimension(tensor_shape.dimension_value(
        self._tensors[0].get_shape()[0]))
    for t in self._tensors[1:]:
      batch_dim.assert_is_compatible_with(tensor_shape.Dimension(
          tensor_shape.dimension_value(t.get_shape()[0])))

    variant_tensor = gen_dataset_ops.tensor_slice_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(TensorSliceDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class SparseTensorSliceDataset(DatasetSource):
  """A `Dataset` that splits a rank-N `tf.sparse.SparseTensor` into its rows."""

  def __init__(self, sparse_tensor):
    """See `Dataset.from_sparse_tensor_slices()` for details."""
    if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
      raise TypeError(
          "`sparse_tensor` must be a `tf.sparse.SparseTensor` object."
          "Was {}.".format(sparse_tensor))
    self._sparse_tensor = sparse_tensor

    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
    self._structure = (tensor_spec.TensorSpec([None, rank], dtypes.int64),
                       tensor_spec.TensorSpec([None],
                                              self._sparse_tensor.dtype),
                       tensor_spec.TensorSpec([rank], dtypes.int64))

    variant_tensor = gen_dataset_ops.sparse_tensor_slice_dataset(
        self._sparse_tensor.indices, self._sparse_tensor.values,
        self._sparse_tensor.dense_shape)
    super(SparseTensorSliceDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _VariantDataset(DatasetV2):
  """A Dataset wrapper around a `tf.variant`-typed function argument."""

  def __init__(self, dataset_variant, structure):
    self._structure = structure
    super(_VariantDataset, self).__init__(dataset_variant)

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._structure


class _NestedVariant(composite_tensor.CompositeTensor):

  def __init__(self, variant_tensor, element_spec, dataset_shape):
    self._variant_tensor = variant_tensor
    self._element_spec = element_spec
    self._dataset_shape = dataset_shape

  @property
  def _type_spec(self):
    return DatasetSpec(self._element_spec, self._dataset_shape)


@tf_export("data.experimental.from_variant")
def from_variant(variant, structure):
  """Constructs a dataset from the given variant and (nested) structure.

  Args:
    variant: A scalar `tf.variant` tensor representing a dataset.
    structure: A (nested) structure of `tf.TypeSpec` objects representing the
      structure of each element in the dataset.

  Returns:
    A `tf.data.Dataset` instance.
  """
  return _VariantDataset(variant, structure)  # pylint: disable=protected-access


@tf_export("data.experimental.to_variant")
def to_variant(dataset):
  """Returns a variant representing the given dataset.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A scalar `tf.variant` tensor representing the given dataset.
  """
  return dataset._variant_tensor  # pylint: disable=protected-access


@tf_export(
    "data.DatasetSpec",
    v1=["data.DatasetSpec", "data.experimental.DatasetStructure"])
class DatasetSpec(type_spec.BatchableTypeSpec):
  """Type specification for `tf.data.Dataset`.

  See `tf.TypeSpec` for more information about TensorFlow type specifications.

  >>> dataset = tf.data.Dataset.range(3)
  >>> tf.data.DatasetSpec.from_value(dataset)
  DatasetSpec(TensorSpec(shape=(), dtype=tf.int64, name=None), TensorShape([]))
  """

  __slots__ = ["_element_spec", "_dataset_shape"]

  def __init__(self, element_spec, dataset_shape=()):
    self._element_spec = element_spec
    self._dataset_shape = tensor_shape.as_shape(dataset_shape)

  @property
  def value_type(self):
    return Dataset

  @property
  def element_spec(self):
    """The inner element spec."""
    return self._element_spec

  def _serialize(self):
    return (self._element_spec, self._dataset_shape)

  @property
  def _component_specs(self):
    return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)

  def _to_components(self, value):
    return value._variant_tensor  # pylint: disable=protected-access

  def _from_components(self, components):
    # pylint: disable=protected-access
    if self._dataset_shape.ndims == 0:
      return _VariantDataset(components, self._element_spec)
    else:
      return _NestedVariant(components, self._element_spec, self._dataset_shape)

  def _to_tensor_list(self, value):
    return [
        ops.convert_to_tensor(
            tf_nest.map_structure(lambda x: x._variant_tensor, value))  # pylint: disable=protected-access
    ]

  @staticmethod
  def from_value(value):
    """Creates a `DatasetSpec` for the given `tf.data.Dataset` value."""
    return DatasetSpec(value.element_spec)  # pylint: disable=protected-access

  def _batch(self, batch_size):
    return DatasetSpec(
        self._element_spec,
        tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))

  def _unbatch(self):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Unbatching a dataset is only supported for rank >= 1")
    return DatasetSpec(self._element_spec, self._dataset_shape[1:])

  def _to_batched_tensor_list(self, value):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Unbatching a dataset is only supported for rank >= 1")
    return self._to_tensor_list(value)

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self


class StructuredFunctionWrapper(object):
  """A function wrapper that supports structured arguments and return values."""

  def __init__(self,
               func,
               transformation_name,
               dataset=None,
               input_classes=None,
               input_shapes=None,
               input_types=None,
               input_structure=None,
               add_to_graph=True,
               use_legacy_function=False,
               defun_kwargs=None):
    """Creates a new `StructuredFunctionWrapper` for the given function.

    Args:
      func: A function from a (nested) structure to another (nested) structure.
      transformation_name: Human-readable name of the transformation in which
        this function is being instantiated, for error messages.
      dataset: (Optional.) A `tf.data.Dataset`. If given, the structure of this
        dataset will be assumed as the structure for `func` arguments; otherwise
        `input_classes`, `input_shapes`, and `input_types` must be defined.
      input_classes: (Optional.) A (nested) structure of `type`. If given, this
        argument defines the Python types for `func` arguments.
      input_shapes: (Optional.) A (nested) structure of `tf.TensorShape`. If
        given, this argument defines the shapes and structure for `func`
        arguments.
      input_types: (Optional.) A (nested) structure of `tf.DType`. If given,
        this argument defines the element types and structure for `func`
        arguments.
      input_structure: (Optional.) A `Structure` object. If given, this argument
        defines the element types and structure for `func` arguments.
      add_to_graph: (Optional.) If `True`, the function will be added to the
        default graph, if it exists.
      use_legacy_function: (Optional.) A boolean that determines whether the
        function be created using `tensorflow.python.eager.function.defun`
        (default behavior) or `tensorflow.python.framework.function.Defun`
        (legacy behavior).
      defun_kwargs: (Optional.) A dictionary mapping string argument names to
        values. If supplied, will be passed to `function` as keyword arguments.

    Raises:
      ValueError: If an invalid combination of `dataset`, `input_classes`,
        `input_shapes`, and `input_types` is passed.
    """
    # pylint: disable=protected-access
    if input_structure is None:
      if dataset is None:
        if input_classes is None or input_shapes is None or input_types is None:
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = structure.convert_legacy_structure(
            input_types, input_shapes, input_classes)
      else:
        if not (input_classes is None and input_shapes is None and
                input_types is None):
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = dataset.element_spec
    else:
      if not (dataset is None and input_classes is None and input_shapes is None
              and input_types is None):
        raise ValueError("Either `dataset`, `input_structure`, or all of "
                         "`input_classes`, `input_shapes`, and `input_types` "
                         "must be specified.")
      self._input_structure = input_structure

    self._func = func

    if defun_kwargs is None:
      defun_kwargs = {}

    readable_transformation_name = transformation_name.replace(
        ".", "_")[:-2] if len(transformation_name) > 2 else ""

    func_name = "_".join(
        [readable_transformation_name,
         function_utils.get_func_name(func)])
    # Sanitize function name to remove symbols that interfere with graph
    # construction.
    for symbol in ["<", ">", "\\", "'", " "]:
      func_name = func_name.replace(symbol, "")

    ag_ctx = autograph_ctx.control_status_ctx()

    def wrapper_helper(*args):
      """Wrapper for passing nested structures to and from tf.data functions."""
      nested_args = structure.from_compatible_tensor_list(
          self._input_structure, args)
      if not _should_unpack(nested_args):
        nested_args = (nested_args,)
      ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
      if _should_pack(ret):
        ret = tuple(ret)

      try:
        self._output_structure = structure.type_spec_from_value(ret)
      except (ValueError, TypeError):
        six.reraise(
            TypeError,
            TypeError("Unsupported return value from function passed to "
                      "%s: %s." % (transformation_name, ret)),
            sys.exc_info()[2])
      return ret

    def trace_legacy_function(defun_kwargs):
      @function.Defun(*structure.get_flat_tensor_types(self._input_structure),
                      **defun_kwargs)
      def wrapped_fn(*args):
        ret = wrapper_helper(*args)
        return structure.to_tensor_list(self._output_structure, ret)

      return lambda: wrapped_fn

    def trace_py_function(defun_kwargs):
      # First we trace the function to infer the output structure.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def unused(*args):  # pylint: disable=missing-docstring,unused-variable
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      _ = unused.get_concrete_function()

      def py_function_wrapper(*args):
        nested_args = structure.from_compatible_tensor_list(
            self._input_structure, args)
        if not _should_unpack(nested_args):
          nested_args = (nested_args,)
        ret = self._func(*nested_args)
        if _should_pack(ret):
          ret = tuple(ret)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      # Next we trace the function wrapped in `eager_py_func` to force eager
      # execution.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def wrapped_fn(*args):  # pylint: disable=missing-docstring
        return script_ops.eager_py_func(
            py_function_wrapper, args,
            structure.get_flat_tensor_types(self._output_structure))

      return wrapped_fn.get_concrete_function

    def trace_tf_function(defun_kwargs):
      # Note: wrapper_helper will apply autograph based on context.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def wrapped_fn(*args):  # pylint: disable=missing-docstring
        ret = wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      return wrapped_fn.get_concrete_function

    if use_legacy_function:
      defun_kwargs.update({"func_name": func_name + "_" + str(ops.uid())})
      fn_factory = trace_legacy_function(defun_kwargs)
    else:
      defun_kwargs.update({"func_name": func_name})
      defun_kwargs.update({"_tf_data_function": True})
      if DEBUG_MODE:
        fn_factory = trace_py_function(defun_kwargs)
      else:
        if def_function.functions_run_eagerly():
          warnings.warn(
              "Even though the `tf.config.experimental_run_functions_eagerly` "
              "option is set, this option does not apply to tf.data functions. "
              "To force eager execution of tf.data functions, please use "
              "`tf.data.experimental.enable_debug_mode()`.")
        fn_factory = trace_tf_function(defun_kwargs)

    self._function = fn_factory()
    # There is no graph to add in eager mode.
    add_to_graph &= not context.executing_eagerly()
    # There are some lifetime issues when a legacy function is not added to a
    # out-living graph. It's already deprecated so de-prioritizing the fix.
    add_to_graph |= use_legacy_function
    if add_to_graph:
      self._function.add_to_graph(ops.get_default_graph())

    if not use_legacy_function:
      outer_graph_seed = ops.get_default_graph().seed
      if outer_graph_seed and self._function.graph.seed == outer_graph_seed:
        if self._function.graph._seed_used:
          warnings.warn(
              "Seed %s from outer graph might be getting used by function %s, "
              "if the random op has not been provided any seed. Explicitly set "
              "the seed in the function if this is not the intended behavior."
              %(outer_graph_seed, func_name), stacklevel=4)

  @property
  def output_structure(self):
    return self._output_structure

  @property
  def output_classes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_shapes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_types(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def function(self):
    return self._function


class _GeneratorDataset(DatasetSource):
  """A `Dataset` that generates elements by invoking a function."""

  def __init__(self, init_args, init_func, next_func, finalize_func,
               output_signature):
    """Constructs a `_GeneratorDataset`.

    Args:
      init_args: A (nested) structure representing the arguments to `init_func`.
      init_func: A TensorFlow function that will be called on `init_args` each
        time a C++ iterator over this dataset is constructed. Returns a (nested)
        structure representing the "state" of the dataset.
      next_func: A TensorFlow function that will be called on the result of
        `init_func` to produce each element, and that raises `OutOfRangeError`
        to terminate iteration.
      finalize_func: A TensorFlow function that will be called on the result of
        `init_func` immediately before a C++ iterator over this dataset is
        destroyed. The return value is ignored.
      output_signature: A (nested) structure of `tf.TypeSpec` objects describing
        the output of `next_func`.
    """
    self._init_args = init_args

    self._init_structure = structure.type_spec_from_value(init_args)

    self._init_func = StructuredFunctionWrapper(
        init_func,
        self._transformation_name(),
        input_structure=self._init_structure)

    self._next_func = StructuredFunctionWrapper(
        next_func,
        self._transformation_name(),
        input_structure=self._init_func.output_structure)

    self._finalize_func = StructuredFunctionWrapper(
        finalize_func,
        self._transformation_name(),
        input_structure=self._init_func.output_structure)

    self._output_signature = output_signature

    variant_tensor = gen_dataset_ops.generator_dataset(
        structure.to_tensor_list(self._init_structure, self._init_args) +
        self._init_func.function.captured_inputs,
        self._next_func.function.captured_inputs,
        self._finalize_func.function.captured_inputs,
        init_func=self._init_func.function,
        next_func=self._next_func.function,
        finalize_func=self._finalize_func.function,
        **self._flat_structure)
    super(_GeneratorDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._output_signature

  def _transformation_name(self):
    return "Dataset.from_generator()"


class ZipDataset(DatasetV2):
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets):
    """See `Dataset.zip()` for details."""
    for ds in nest.flatten(datasets):
      if not isinstance(ds, DatasetV2):
        if isinstance(ds, list):
          message = ("The argument to `Dataset.zip()` must be a (nested) "
                     "structure of `Dataset` objects. Python `list` is not "
                     "supported, please use a `tuple` instead.")
        else:
          message = ("The argument to `Dataset.zip()` must be a (nested) "
                     "structure of `Dataset` objects.")
        raise TypeError(message)
    self._datasets = datasets
    self._structure = nest.pack_sequence_as(
        self._datasets,
        [ds.element_spec for ds in nest.flatten(self._datasets)])
    variant_tensor = gen_dataset_ops.zip_dataset(
        [ds._variant_tensor for ds in nest.flatten(self._datasets)],
        **self._flat_structure)
    super(ZipDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return nest.flatten(self._datasets)

  @property
  def element_spec(self):
    return self._structure


class ConcatenateDataset(DatasetV2):
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate):
    """See `Dataset.concatenate()` for details."""
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate

    output_types = get_legacy_output_types(input_dataset)
    if output_types != get_legacy_output_types(dataset_to_concatenate):
      raise TypeError(
          "Two datasets to concatenate have different types %s and %s" %
          (output_types, get_legacy_output_types(dataset_to_concatenate)))

    output_classes = get_legacy_output_classes(input_dataset)
    if output_classes != get_legacy_output_classes(dataset_to_concatenate):
      raise TypeError(
          "Two datasets to concatenate have different classes %s and %s" %
          (output_classes, get_legacy_output_classes(dataset_to_concatenate)))

    spec1 = input_dataset.element_spec
    spec2 = dataset_to_concatenate.element_spec
    self._structure = nest.pack_sequence_as(spec1, [
        ts1.most_specific_compatible_type(ts2)
        for (ts1, ts2) in zip(nest.flatten(spec1), nest.flatten(spec2))
    ])

    self._input_datasets = [input_dataset, dataset_to_concatenate]
    # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.concatenate_dataset(
        input_dataset._variant_tensor, dataset_to_concatenate._variant_tensor,
        **self._flat_structure)
    # pylint: enable=protected-access
    super(ConcatenateDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return self._input_datasets

  @property
  def element_spec(self):
    return self._structure


class RepeatDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count):
    """See `Dataset.repeat()` for details."""
    self._input_dataset = input_dataset
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.repeat_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(RepeatDataset, self).__init__(input_dataset, variant_tensor)


class RangeDataset(DatasetSource):
  """A `Dataset` of a step separated range of values."""

  def __init__(self, *args, **kwargs):
    """See `Dataset.range()` for details."""
    self._parse_args(*args, **kwargs)
    self._structure = tensor_spec.TensorSpec([], self._output_type)
    variant_tensor = gen_dataset_ops.range_dataset(
        start=self._start,
        stop=self._stop,
        step=self._step,
        **self._flat_structure)
    super(RangeDataset, self).__init__(variant_tensor)

  def _parse_args(self, *args, **kwargs):
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
    if "output_type" in kwargs:
      self._output_type = kwargs["output_type"]
    else:
      self._output_type = dtypes.int64

  def _build_tensor(self, int64_value, name):
    return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

  @property
  def element_spec(self):
    return self._structure


class CacheDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename):
    """See `Dataset.cache()` for details."""
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")
    if tf2.enabled() and (context.executing_eagerly() or ops.inside_function()):
      variant_tensor = gen_dataset_ops.cache_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          cache=gen_dataset_ops.dummy_memory_cache(),
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.cache_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          **self._flat_structure)
    super(CacheDataset, self).__init__(input_dataset, variant_tensor)


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
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if invalid arguments are provided.
    """
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._seed, self._seed2 = random_seed.get_seed(seed)
    if reshuffle_each_iteration is None:
      reshuffle_each_iteration = True
    self._reshuffle_each_iteration = reshuffle_each_iteration

    if (tf2.enabled() and
        (context.executing_eagerly() or ops.inside_function())):
      variant_tensor = gen_dataset_ops.shuffle_dataset_v3(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed=self._seed,
          seed2=self._seed2,
          seed_generator=gen_dataset_ops.dummy_seed_generator(),
          reshuffle_each_iteration=self._reshuffle_each_iteration,
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.shuffle_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed=self._seed,
          seed2=self._seed2,
          reshuffle_each_iteration=self._reshuffle_each_iteration,
          **self._flat_structure)
    super(ShuffleDataset, self).__init__(input_dataset, variant_tensor)


class TakeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` containing the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.take()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.take_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(TakeDataset, self).__init__(input_dataset, variant_tensor)


class SkipDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.skip()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.skip_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(SkipDataset, self).__init__(input_dataset, variant_tensor)


class ShardDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` for sharding its input."""

  def __init__(self, input_dataset, num_shards, index):
    """See `Dataset.shard()` for details."""
    self._input_dataset = input_dataset
    self._num_shards = ops.convert_to_tensor(
        num_shards, dtype=dtypes.int64, name="num_shards")
    self._index = ops.convert_to_tensor(index, dtype=dtypes.int64, name="index")
    variant_tensor = gen_dataset_ops.shard_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_shards=self._num_shards,
        index=self._index,
        **self._flat_structure)
    super(ShardDataset, self).__init__(input_dataset, variant_tensor)


class BatchDataset(UnaryDataset):
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, drop_remainder):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      constant_batch_size = tensor_util.constant_value(self._batch_size)
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(constant_batch_size),
          input_dataset.element_spec)
    else:
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          input_dataset.element_spec)
    variant_tensor = gen_dataset_ops.batch_dataset_v2(
        input_dataset._variant_tensor,
        batch_size=self._batch_size,
        drop_remainder=self._drop_remainder,
        **self._flat_structure)
    super(BatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class ParallelBatchDataset(UnaryDataset):
  """A `Dataset` that batches contiguous elements from its input in parallel."""

  def __init__(self, input_dataset, batch_size, drop_remainder,
               num_parallel_calls, deterministic):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      constant_batch_size = tensor_util.constant_value(self._batch_size)
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(constant_batch_size),
          input_dataset.element_spec)
    else:
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          input_dataset.element_spec)

    variant_tensor = gen_dataset_ops.parallel_batch_dataset(
        input_dataset._variant_tensor,
        batch_size=self._batch_size,
        num_parallel_calls=self._num_parallel_calls,
        drop_remainder=self._drop_remainder,
        deterministic=self._deterministic,
        **self._flat_structure)

    super(ParallelBatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _NumpyIterator(object):
  """Iterator over a dataset with elements converted to numpy."""

  __slots__ = ["_iterator"]

  def __init__(self, dataset):
    self._iterator = iter(dataset)

  def __iter__(self):
    return self

  def __next__(self):

    def to_numpy(x):
      numpy = x._numpy()  # pylint: disable=protected-access
      if isinstance(numpy, np.ndarray):
        # `numpy` shares the same underlying buffer as the `x` Tensor.
        # Tensors are expected to be immutable, so we disable writes.
        numpy.setflags(write=False)
      return numpy

    return nest.map_structure(to_numpy, next(self._iterator))

  def next(self):
    return self.__next__()


class _VariantTracker(tracking.CapturableResource):
  """Allows export of functions capturing a Dataset in SavedModels.

  When saving a SavedModel, `tf.saved_model.save` traverses the object
  graph. Since Datasets reference _VariantTracker objects, that traversal will
  find a _VariantTracker for each Dataset and so know how to save and restore
  functions which reference the Dataset's variant Tensor.
  """

  def __init__(self, variant_tensor, resource_creator):
    """Record that `variant_tensor` is associated with `resource_creator`.

    Args:
      variant_tensor: The variant-dtype Tensor associated with the Dataset. This
        Tensor will be a captured input to functions which use the Dataset, and
        is used by saving code to identify the corresponding _VariantTracker.
      resource_creator: A zero-argument function which creates a new
        variant-dtype Tensor. This function will be included in SavedModels and
        run to re-create the Dataset's variant Tensor on restore.
    """
    super(_VariantTracker, self).__init__(device="CPU")
    self._resource_handle = variant_tensor
    self._create_resource = resource_creator


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
      six.reraise(ValueError, ValueError(
          "Padded shape %s must be a 1-D tensor of tf.int64 values, but its "
          "shape was %s." % (padded_shape, ret.shape)), sys.exc_info()[2])
    if ret.dtype != dtypes.int64:
      six.reraise(
          TypeError,
          TypeError(
              "Padded shape %s must be a 1-D tensor of tf.int64 values, but "
              "its element type was %s." % (padded_shape, ret.dtype.name)),
          sys.exc_info()[2])
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
  if not value.shape.is_compatible_with(tensor_shape.TensorShape([])):
    raise ValueError("Padding value should be a scalar, but is not: %s" % value)
  if value.dtype != output_type:
    raise TypeError("Padding value tensor (%s) does not match output type: %s" %
                    (value, output_type))
  return value


def _padding_values_or_default(padding_values, input_dataset):
  """Returns padding values with None elements replaced with default values."""

  def make_zero(t):
    if t.base_dtype == dtypes.string:
      return ""
    elif t.base_dtype == dtypes.variant:
      error_msg = ("Unable to create padding for field of type 'variant' "
                   "because t.base_type == dtypes.variant == "
                   "{}.".format(t.base_dtype))
      raise TypeError(error_msg)
    elif t.base_dtype == dtypes.bfloat16:
      # Special case `bfloat16` because it is not supported by NumPy.
      return constant_op.constant(0, dtype=dtypes.bfloat16)
    else:
      return np.zeros_like(t.as_numpy_dtype())

  def value_or_default(value, default):
    return default if value is None else value

  default_padding = nest.map_structure(
      make_zero,
      get_legacy_output_types(input_dataset))
  return nest.map_structure_up_to(padding_values, value_or_default,
                                  padding_values, default_padding)


class PaddedBatchDataset(UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values,
               drop_remainder):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset

    def check_types(component_spec):
      if not isinstance(component_spec, tensor_spec.TensorSpec):
        raise TypeError("Padded batching of components of type ",
                        type(component_spec), " is not supported.")

    nest.map_structure(check_types, input_dataset.element_spec)
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    padding_values = _padding_values_or_default(padding_values, input_dataset)

    input_shapes = get_legacy_output_shapes(input_dataset)
    flat_padded_shapes = nest.flatten_up_to(input_shapes, padded_shapes)

    flat_padded_shapes_as_tensors = []

    for input_component_shape, padded_shape in zip(
        nest.flatten(input_shapes), flat_padded_shapes):
      flat_padded_shapes_as_tensors.append(
          _padded_shape_to_tensor(padded_shape, input_component_shape))

    self._padded_shapes = nest.pack_sequence_as(input_shapes,
                                                flat_padded_shapes_as_tensors)

    # If padding_values is a single element and input_shapes is a structure,
    # "broadcast" padding_values to the same structure as input_shapes.
    if nest.is_sequence(input_shapes) and not nest.is_sequence(padding_values):
      padding_values = nest.map_structure(lambda _: padding_values,
                                          input_shapes)

    self._padding_values = nest.map_structure_up_to(
        input_shapes, _padding_value_to_tensor, padding_values,
        get_legacy_output_types(input_dataset))
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    def _padded_shape_to_batch_shape(s):
      return tensor_shape.TensorShape([
          tensor_util.constant_value(self._batch_size)
          if smart_cond.smart_constant_value(self._drop_remainder) else None
      ]).concatenate(tensor_util.constant_value_as_shape(s))

    output_shapes = nest.map_structure(
        _padded_shape_to_batch_shape, self._padded_shapes)
    self._structure = structure.convert_legacy_structure(
        get_legacy_output_types(self._input_dataset), output_shapes,
        get_legacy_output_classes(self._input_dataset))

    # pylint: disable=protected-access
    # TODO(jsimsa): Switch to using v2 only any time after 6/30/2018.
    if smart_cond.smart_constant_value(self._drop_remainder) is False:
      variant_tensor = gen_dataset_ops.padded_batch_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          output_shapes=structure.get_flat_tensor_shapes(self._structure))
    else:
      variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          drop_remainder=self._drop_remainder,
          output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(PaddedBatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


def _should_pack(arg):
  """Determines whether the caller needs to pack the argument in a tuple.

  If user-defined function returns a list of tensors, `nest.flatten()` and
  `ops.convert_to_tensor()` and would conspire to attempt to stack those tensors
  into a single tensor because the tf.data version of `nest.flatten()` does
  not recurse into lists. Since it is more likely that the list arose from
  returning the result of an operation (such as `tf.numpy_function()`) that
  returns a list of not-necessarily-stackable tensors, we treat the returned
  value as a `tuple` instead. A user wishing to pack the return value into a
  single tensor can use an explicit `tf.stack()` before returning.

  Args:
    arg: argument to check

  Returns:
    Indication of whether the caller needs to pack the argument in a tuple.
  """
  return isinstance(arg, list)


def _should_unpack(arg):
  """Determines whether the caller needs to unpack the argument from a tuple.

  Args:
    arg: argument to check

  Returns:
    Indication of whether the caller needs to unpack the argument from a tuple.
  """
  return type(arg) is tuple  # pylint: disable=unidiomatic-typecheck


class MapDataset(UnaryDataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self,
               input_dataset,
               map_func,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality
    self._map_func = StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(MapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class ParallelMapDataset(UnaryDataset):
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               deterministic,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if deterministic is None:
      self._deterministic = "default"
    elif deterministic:
      self._deterministic = "true"
    else:
      self._deterministic = "false"
    self._preserve_cardinality = preserve_cardinality
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    variant_tensor = gen_dataset_ops.parallel_map_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        num_parallel_calls=self._num_parallel_calls,
        deterministic=self._deterministic,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(ParallelMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class FlatMapDataset(UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(FlatMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.flat_map()"


class InterleaveDataset(UnaryDataset):
  """A `Dataset` that interleaves the result of transformed inputs."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length):
    """See `Dataset.interleave()` for details."""

    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")

    variant_tensor = gen_dataset_ops.interleave_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        f=self._map_func.function,
        **self._flat_structure)
    super(InterleaveDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"


class ParallelInterleaveDataset(UnaryDataset):
  """A `Dataset` that maps a function over its input and interleaves the result."""

  def __init__(self,
               input_dataset,
               map_func,
               cycle_length,
               block_length,
               num_parallel_calls,
               buffer_output_elements=AUTOTUNE,
               prefetch_input_elements=AUTOTUNE,
               deterministic=None):
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._buffer_output_elements = ops.convert_to_tensor(
        buffer_output_elements,
        dtype=dtypes.int64,
        name="buffer_output_elements")
    self._prefetch_input_elements = ops.convert_to_tensor(
        prefetch_input_elements,
        dtype=dtypes.int64,
        name="prefetch_input_elements")

    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    if deterministic is None:
      deterministic_string = "default"
    elif deterministic:
      deterministic_string = "true"
    else:
      deterministic_string = "false"

    variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v4(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        self._buffer_output_elements,
        self._prefetch_input_elements,
        self._num_parallel_calls,
        f=self._map_func.function,
        deterministic=deterministic_string,
        **self._flat_structure)
    super(ParallelInterleaveDataset, self).__init__(input_dataset,
                                                    variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"


class FilterDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self, input_dataset, predicate, use_legacy_function=False):
    """See `Dataset.filter()` for details."""
    self._input_dataset = input_dataset
    wrapped_func = StructuredFunctionWrapper(
        predicate,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      error_msg = ("`predicate` return type must be convertible to a scalar "
                   "boolean tensor. Was {}.").format(
                       wrapped_func.output_structure)
      raise ValueError(error_msg)
    self._predicate = wrapped_func
    variant_tensor = gen_dataset_ops.filter_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._flat_structure)
    super(FilterDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._predicate]

  def _transformation_name(self):
    return "Dataset.filter()"


class PrefetchDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size, slack_period=None):
    """See `Dataset.prefetch()` for details.

    Args:
      input_dataset: The input dataset.
      buffer_size: See `Dataset.prefetch()` for details.
      slack_period: (Optional.) An integer. If non-zero, determines the number
        of GetNext calls before injecting slack into the execution. This may
        reduce CPU contention at the start of a step. Note that a tensorflow
        user should not have to set this manually; enable this behavior
        automatically via `tf.data.Options.experimental_slack` instead. Defaults
        to None.
    """
    self._input_dataset = input_dataset
    if buffer_size is None:
      buffer_size = AUTOTUNE
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    # pylint: disable=protected-access
    # We colocate the prefetch dataset with its input as this collocation only
    # happens automatically in graph mode.
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.prefetch_dataset(
          input_dataset._variant_tensor,
          buffer_size=self._buffer_size,
          slack_period=slack_period,
          **self._flat_structure)
    super(PrefetchDataset, self).__init__(input_dataset, variant_tensor)


class WindowDataset(UnaryDataset):
  """A dataset that creates window datasets from the input elements."""

  def __init__(self, input_dataset, size, shift, stride, drop_remainder):
    """See `window_dataset()` for more details."""
    self._input_dataset = input_dataset
    self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name="size")
    self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name="shift")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._structure = nest.pack_sequence_as(
        get_legacy_output_classes(input_dataset), [
            DatasetSpec(  # pylint: disable=g-complex-comprehension
                structure.convert_legacy_structure(
                    output_type, output_shape, output_class))
            for output_class, output_shape, output_type in zip(
                nest.flatten(get_legacy_output_classes(input_dataset)),
                nest.flatten(get_legacy_output_shapes(input_dataset)),
                nest.flatten(get_legacy_output_types(input_dataset)))
        ])
    variant_tensor = gen_dataset_ops.window_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._size,
        self._shift,
        self._stride,
        self._drop_remainder,
        **self._flat_structure)
    super(WindowDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _OptionsDataset(UnaryUnchangedStructureDataset):
  """An identity `Dataset` that stores options."""

  def __init__(self, input_dataset, options):
    # pylint: disable=protected-access
    self._input_dataset = input_dataset
    options_pb = dataset_options_pb2.Options()
    options_pb.CopyFrom(options._to_proto())
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.options_dataset(
          input_dataset._variant_tensor,
          options_pb.SerializeToString(), **self._flat_structure)
    super(_OptionsDataset, self).__init__(input_dataset, variant_tensor)

    if self._options_attr:
      self._options_attr._set_mutable(True)
      self._options_attr = self._options_attr.merge(options)
    else:
      self._options_attr = options
    self._options_attr._set_mutable(False)


def normalize_to_dense(dataset):
  """Normalizes non-tensor components in a dataset to dense representations.

  This is necessary for dataset transformations that slice along the batch
  dimension and are oblivious to non-tensors, e.g. `unbatch`, `rebatch`.

  Args:
    dataset: Dataset to normalize.

  Returns:
    A dataset whose sparse and ragged tensors have been normalized to their
    dense representations.
  """

  # NOTE(mrry): This leads to a somewhat inefficient re-encoding step for all
  # non-tensor components.
  #
  # TODO(mrry): Consider optimizing this if it turns out to be a bottleneck.
  if _should_unpack(dataset.element_spec):

    def normalize(*args):
      return structure.to_batched_tensor_list(dataset.element_spec, tuple(args))
  else:
    def normalize(arg):
      return structure.to_batched_tensor_list(dataset.element_spec, arg)

  normalized_dataset = dataset.map(normalize)

  # NOTE(mrry): Our `map()` has lost information about the structure of
  # non-tensor components, so re-apply the structure of the original dataset.
  return _RestructuredDataset(normalized_dataset, dataset.element_spec)


class _RestructuredDataset(UnaryDataset):
  """An internal helper for changing the element spec of a dataset."""

  def __init__(self, dataset, structure):
    self._input_dataset = dataset
    self._structure = structure

    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_RestructuredDataset, self).__init__(dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _UnbatchDataset(UnaryDataset):
  """A dataset that splits the elements of its input into multiple elements."""

  def __init__(self, input_dataset):
    """See `unbatch()` for more details."""
    flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
    if any(s.ndims == 0 for s in flat_shapes):
      raise ValueError("Cannot unbatch an input with scalar components.")
    known_batch_dim = tensor_shape.Dimension(None)
    for s in flat_shapes:
      try:
        known_batch_dim = known_batch_dim.merge_with(s[0])
      except ValueError:
        raise ValueError("Cannot unbatch an input whose components have "
                         "different batch sizes.")
    self._input_dataset = input_dataset
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(),  # pylint: disable=protected-access
        get_structure(input_dataset))
    variant_tensor = ged_ops.unbatch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._flat_structure)
    super(_UnbatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _GroupByWindowDataset(UnaryDataset):
  """A `Dataset` that groups its input and performs a windowed reduction."""

  def __init__(self, input_dataset, key_func, reduce_func, window_size_func):
    """See `group_by_window()` for details."""
    self._input_dataset = input_dataset
    self._make_key_func(key_func, input_dataset)
    self._make_reduce_func(reduce_func, input_dataset)
    self._make_window_size_func(window_size_func)
    variant_tensor = ged_ops.group_by_window_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._window_size_func.function.captured_inputs,
        key_func=self._key_func.function,
        reduce_func=self._reduce_func.function,
        window_size_func=self._window_size_func.function,
        **self._flat_structure)
    super(_GroupByWindowDataset, self).__init__(input_dataset, variant_tensor)

  def _make_window_size_func(self, window_size_func):
    """Make wrapping defun for window_size_func."""

    def window_size_func_wrapper(key):
      return ops.convert_to_tensor(window_size_func(key), dtype=dtypes.int64)

    self._window_size_func = StructuredFunctionWrapper(
        window_size_func_wrapper,
        self._transformation_name(),
        input_structure=tensor_spec.TensorSpec([], dtypes.int64))
    if not self._window_size_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(
          "`window_size_func` must return a single tf.int64 scalar tensor.")

  def _make_key_func(self, key_func, input_dataset):
    """Make wrapping defun for key_func."""

    def key_func_wrapper(*args):
      return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)

    self._key_func = StructuredFunctionWrapper(
        key_func_wrapper, self._transformation_name(), dataset=input_dataset)
    if not self._key_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(
          "`key_func` must return a single tf.int64 scalar tensor.")

  def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping defun for reduce_func."""
    nested_dataset = DatasetSpec(input_dataset.element_spec)
    input_structure = (tensor_spec.TensorSpec([], dtypes.int64), nested_dataset)
    self._reduce_func = StructuredFunctionWrapper(
        reduce_func,
        self._transformation_name(),
        input_structure=input_structure)
    if not isinstance(self._reduce_func.output_structure, DatasetSpec):
      raise TypeError("`reduce_func` must return a `Dataset` object.")
    # pylint: disable=protected-access
    self._element_spec = (self._reduce_func.output_structure._element_spec)

  @property
  def element_spec(self):
    return self._element_spec

  def _functions(self):
    return [self._key_func, self._reduce_func, self._window_size_func]

  def _transformation_name(self):
    return "Dataset.group_by_window()"


class RandomDataset(DatasetSource):
  """A `Dataset` of pseudorandom values."""

  def __init__(self, seed=None):
    """A `Dataset` of pseudorandom values."""
    self._seed, self._seed2 = random_seed.get_seed(seed)
    variant_tensor = ged_ops.random_dataset(
        seed=self._seed, seed2=self._seed2, **self._flat_structure)
    super(RandomDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.int64)


class _TakeWhileDataset(UnaryUnchangedStructureDataset):
  """A dataset that stops iteration when `predicate` returns false."""

  def __init__(self, input_dataset, predicate):
    """See `take_while()` for details."""

    self._input_dataset = input_dataset
    wrapped_func = StructuredFunctionWrapper(
        predicate, self._transformation_name(), dataset=self._input_dataset)

    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      raise ValueError("`predicate` must return a scalar boolean tensor.")

    self._predicate = wrapped_func
    var_tensor = ged_ops.take_while_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._flat_structure)
    super(_TakeWhileDataset, self).__init__(input_dataset, var_tensor)

  def _functions(self):
    return [self._predicate]

  def _transformation_name(self):
    return "Dataset.take_while()"


class _UniqueDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` contains the unique elements from its input."""

  def __init__(self, input_dataset):
    """See `unique()` for details."""
    self._input_dataset = input_dataset
    if get_legacy_output_types(input_dataset) not in (dtypes.int32,
                                                      dtypes.int64,
                                                      dtypes.string):
      raise TypeError(
          "`tf.data.Dataset.unique()` only supports inputs with a single "
          "`tf.int32`, `tf.int64`, or `tf.string` component.")
    variant_tensor = ged_ops.unique_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._flat_structure)
    super(_UniqueDataset, self).__init__(input_dataset, variant_tensor)


def _collect_resource_inputs(op):
  """Collects resource inputs for the given ops (and its variant inputs)."""

  def _process(op_queue, seen_ops):
    """Processes the next element of the op queue.

    Args:
      op_queue: Queue of Dataset operations to process.
      seen_ops: Already processed set of Operations.

    Returns:
      A 2-tuple containing sets of resource handles. The first tuple entry
      contains read-only handles and the second entry contains read-write
      handles.
    """

    reads = []
    writes = []
    op = op_queue.pop()
    if op in seen_ops:
      return reads, writes
    seen_ops.add(op)
    # TODO(b/150139257): All resource inputs are in writes right now since we
    # have not updated the functional ops to set the special attribute that ACD
    # uses to figure out which of the op's inputs are read-only.
    reads, writes = acd_utils.get_read_write_resource_inputs(op)
    # Conservatively assume that any variant inputs are datasets.
    op_queue.extend(t.op for t in op.inputs if t.dtype == dtypes.variant)
    return reads, writes

  op_queue = [op]
  seen_ops = set()
  all_reads = []
  all_writes = []
  while op_queue:
    reads, writes = _process(op_queue, seen_ops)
    all_reads.extend(reads)
    all_writes.extend(writes)

  return all_reads, all_writes


class _SnapshotDataset(UnaryUnchangedStructureDataset):
  """A dataset that allows saving and re-use of already processed data."""

  def __init__(self,
               input_dataset,
               path,
               shard_func,
               compression=None,
               reader_func=None,
               pending_snapshot_expiry_seconds=None,
               use_legacy_function=False):

    if reader_func is None:
      reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
          lambda x: x,
          cycle_length=multiprocessing.cpu_count(),
          num_parallel_calls=AUTOTUNE)

    self._input_dataset = input_dataset
    self._path = path
    self._compression = compression

    self._reader_func = StructuredFunctionWrapper(
        reader_func,
        self._transformation_name() + ".reader_func",
        # Dataset of datasets of input elements
        input_structure=DatasetSpec(DatasetSpec(input_dataset.element_spec)),
        use_legacy_function=use_legacy_function)
    self._shard_func = StructuredFunctionWrapper(
        shard_func,
        self._transformation_name() + ".shard_func",
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)

    if ((not self._shard_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int32))) and
        (not self._shard_func.output_structure.is_compatible_with(
            tensor_spec.TensorSpec([], dtypes.int64)))):
      raise TypeError(
          "shard_func must return a 0-dimension tensor containing an int.")

    variant_tensor = ged_ops.snapshot_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        path,
        self._reader_func.function.captured_inputs,
        self._shard_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        shard_func=self._shard_func.function,
        **self._flat_structure)
    super(_SnapshotDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._reader_func, self._shard_func]

  def _transformation_name(self):
    return "Dataset.snapshot()"


class _ScanDataset(UnaryDataset):
  """A dataset that scans a function across its input."""

  def __init__(self,
               input_dataset,
               initial_state,
               scan_func,
               use_default_device=None):
    """See `scan()` for details."""
    self._input_dataset = input_dataset
    self._initial_state = structure.normalize_element(initial_state)

    # Compute initial values for the state classes, shapes and types based on
    # the initial state. The shapes may be refined by running `tf_scan_func` one
    # or more times below.
    self._state_structure = structure.type_spec_from_value(self._initial_state)

    # Iteratively rerun the scan function until reaching a fixed point on
    # `self._state_shapes`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = StructuredFunctionWrapper(
          scan_func,
          self._transformation_name(),
          input_structure=(self._state_structure, input_dataset.element_spec),
          add_to_graph=False)
      if not (isinstance(wrapped_func.output_types, collections_abc.Sequence)
              and len(wrapped_func.output_types) == 2):
        raise TypeError("The scan function must return a pair comprising the "
                        "new state and the output value.")

      new_state_classes, self._output_classes = wrapped_func.output_classes

      # Extract and validate class information from the returned values.
      new_state_classes, output_classes = wrapped_func.output_classes
      old_state_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
          self._state_structure)
      for new_state_class, old_state_class in zip(
          nest.flatten(new_state_classes), nest.flatten(old_state_classes)):
        if not issubclass(new_state_class, old_state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (old_state_classes, new_state_classes))

      # Extract and validate type information from the returned values.
      new_state_types, output_types = wrapped_func.output_types
      old_state_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
          self._state_structure)
      for new_state_type, old_state_type in zip(
          nest.flatten(new_state_types), nest.flatten(old_state_types)):
        if new_state_type != old_state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (old_state_types, new_state_types))

      # Extract shape information from the returned values.
      new_state_shapes, output_shapes = wrapped_func.output_shapes
      old_state_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
          self._state_structure)
      self._element_spec = structure.convert_legacy_structure(
          output_types, output_shapes, output_classes)

      flat_state_shapes = nest.flatten(old_state_shapes)
      flat_new_state_shapes = nest.flatten(new_state_shapes)
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
        # TODO(b/110122868): Support a "most specific compatible structure"
        # method for combining structures, to avoid using legacy structures
        # in this method.
        self._state_structure = structure.convert_legacy_structure(
            old_state_types,
            nest.pack_sequence_as(old_state_shapes, weakened_state_shapes),
            old_state_classes)

    self._scan_func = wrapped_func
    self._scan_func.function.add_to_graph(ops.get_default_graph())
    # pylint: disable=protected-access
    if use_default_device is not None:
      variant_tensor = ged_ops.scan_dataset(
          self._input_dataset._variant_tensor,
          structure.to_tensor_list(self._state_structure, self._initial_state),
          self._scan_func.function.captured_inputs,
          f=self._scan_func.function,
          preserve_cardinality=True,
          use_default_device=use_default_device,
          **self._flat_structure)
    else:
      variant_tensor = ged_ops.scan_dataset(
          self._input_dataset._variant_tensor,
          structure.to_tensor_list(self._state_structure, self._initial_state),
          self._scan_func.function.captured_inputs,
          f=self._scan_func.function,
          preserve_cardinality=True,
          **self._flat_structure)
    super(_ScanDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._scan_func]

  @property
  def element_spec(self):
    return self._element_spec

  def _transformation_name(self):
    return "Dataset.scan()"


@auto_control_deps.register_acd_resource_resolver
def _resource_resolver(op, resource_reads, resource_writes):
  """Updates resource inputs for tf.data ops with indirect dependencies."""

  updated = False
  if op.type in [
      "DatasetToSingleElement", "DatasetToTFRecord", "ReduceDataset"
  ]:
    reads, writes = _collect_resource_inputs(op)
    for inp in reads:
      if inp not in resource_reads:
        updated = True
        resource_reads.add(inp)
    for inp in writes:
      if inp not in resource_writes:
        updated = True
        resource_writes.add(inp)

  if op.type in [
      "IteratorGetNext", "IteratorGetNextSync", "IteratorGetNextAsOptional"
  ]:
    iterator_resource = op.inputs[0]
    make_iterator_ops = [
        op for op in iterator_resource.consumers() if op.type == "MakeIterator"
    ]

    if len(make_iterator_ops) == 1:
      reads, writes = _collect_resource_inputs(make_iterator_ops[0])
      for inp in reads:
        if inp not in resource_reads:
          updated = True
          resource_reads.add(inp)
      for inp in writes:
        if inp not in resource_writes:
          updated = True
          resource_writes.add(inp)

  return updated


DEBUG_MODE = False


@tf_export("data.experimental.enable_debug_mode")
def enable_debug_mode():
  """Enables debug mode for tf.data.

  Example usage with pdb module:
  ```
  import tensorflow as tf
  import pdb

  tf.data.experimental.enable_debug_mode()

  def func(x):
    # Python 3.7 and older requires `pdb.Pdb(nosigint=True).set_trace()`
    pdb.set_trace()
    x = x + 1
    return x

  dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  dataset = dataset.map(func)

  for item in dataset:
    print(item)
  ```

  The effect of debug mode is two-fold:

  1) Any transformations that would introduce asynchrony, parallelism, or
  non-determinism to the input pipeline execution will be forced to execute
  synchronously, sequentially, and deterministically.

  2) Any user-defined functions passed into tf.data transformations such as
  `map` will be wrapped in `tf.py_function` so that their body is executed
  "eagerly" as a Python function as opposed to a traced TensorFlow graph, which
  is the default behavior. Note that even when debug mode is enabled, the
  user-defined function is still traced  to infer the shape and type of its
  outputs; as a consequence, any `print` statements or breakpoints will be
  triggered once during the tracing before the actual execution of the input
  pipeline.

  NOTE: As the debug mode setting affects the construction of the tf.data input
  pipeline, it should be enabled before any tf.data definitions.

  Raises:
    ValueError: When invoked from graph mode.
  """
  if context.executing_eagerly():
    toggle_debug_mode(True)
  else:
    raise ValueError("Debug mode is only supported in eager mode.")


def toggle_debug_mode(debug_mode):
  global DEBUG_MODE
  DEBUG_MODE = debug_mode
