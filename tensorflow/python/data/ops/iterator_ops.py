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
"""Python wrappers for Iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import warnings

from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# NOTE(mrry): It is legitimate to call `Iterator.get_next()` multiple
# times, e.g. when you are distributing different elements to multiple
# devices in a single step. However, a common pitfall arises when
# users call `Iterator.get_next()` in each iteration of their training
# loop. `Iterator.get_next()` adds ops to the graph, and executing
# each op allocates resources (including threads); as a consequence,
# invoking it in every iteration of a training loop causes slowdown
# and eventual resource exhaustion. To guard against this outcome, we
# log a warning when the number of uses crosses a threshold of suspicion.
GET_NEXT_CALL_WARNING_THRESHOLD = 32

GET_NEXT_CALL_WARNING_MESSAGE = (
    "An unusually high number of `Iterator.get_next()` calls was detected. "
    "This often indicates that `Iterator.get_next()` is being called inside "
    "a training loop, which will cause gradual slowdown and eventual resource "
    "exhaustion. If this is the case, restructure your code to call "
    "`next_element = iterator.get_next()` once outside the loop, and use "
    "`next_element` as the input to some computation that is invoked inside "
    "the loop.")

# Collection of all IteratorResources in the `Graph`.
GLOBAL_ITERATORS = "iterators"


def _device_stack_is_empty():
  if context.executing_eagerly():
    return context.context().device_name is None
  # pylint: disable=protected-access
  device_stack = ops.get_default_graph()._device_functions_outer_to_inner
  # pylint: enable=protected-access
  return not bool(device_stack)


@tf_export(v1=["data.Iterator"])
class Iterator(trackable.Trackable):
  """Represents the state of iterating through a `Dataset`."""

  def __init__(self, iterator_resource, initializer, output_types,
               output_shapes, output_classes):
    """Creates a new iterator from the given iterator resource.

    Note: Most users will not call this initializer directly, and will
    instead use `Dataset.make_initializable_iterator()` or
    `Dataset.make_one_shot_iterator()`.

    Args:
      iterator_resource: A `tf.resource` scalar `tf.Tensor` representing the
        iterator.
      initializer: A `tf.Operation` that should be run to initialize this
        iterator.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this iterator.
      output_shapes: A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this iterator.
      output_classes: A nested structure of Python `type` objects corresponding
        to each component of an element of this iterator.
    """
    self._iterator_resource = iterator_resource
    self._initializer = initializer

    if (output_types is None or output_shapes is None
        or output_classes is None):
      raise ValueError("If `structure` is not specified, all of "
                       "`output_types`, `output_shapes`, and `output_classes`"
                       " must be specified.")
    self._element_spec = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    self._flat_tensor_shapes = structure.get_flat_tensor_shapes(
        self._element_spec)
    self._flat_tensor_types = structure.get_flat_tensor_types(
        self._element_spec)

    self._string_handle = gen_dataset_ops.iterator_to_string_handle(
        self._iterator_resource)
    self._get_next_call_count = 0
    ops.add_to_collection(GLOBAL_ITERATORS, self._iterator_resource)

  @staticmethod
  def from_structure(output_types,
                     output_shapes=None,
                     shared_name=None,
                     output_classes=None):
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
        each component of an element of this dataset.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset. If
        omitted, each component will have an unconstrainted shape.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).
      output_classes: (Optional.) A nested structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

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
      output_shapes = nest.map_structure_up_to(output_types,
                                               tensor_shape.as_shape,
                                               output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: ops.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    output_structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    if shared_name is None:
      shared_name = ""
    if _device_stack_is_empty():
      with ops.device("/cpu:0"):
        iterator_resource = gen_dataset_ops.iterator_v2(
            container="",
            shared_name=shared_name,
            output_types=structure.get_flat_tensor_types(
                output_structure),
            output_shapes=structure.get_flat_tensor_shapes(
                output_structure))
    else:
      iterator_resource = gen_dataset_ops.iterator_v2(
          container="",
          shared_name=shared_name,
          output_types=structure.get_flat_tensor_types(output_structure),
          output_shapes=structure.get_flat_tensor_shapes(
              output_structure))
    return Iterator(iterator_resource, None, output_types, output_shapes,
                    output_classes)

  @staticmethod
  def from_string_handle(string_handle,
                         output_types,
                         output_shapes=None,
                         output_classes=None):
    """Creates a new, uninitialized `Iterator` based on the given handle.

    This method allows you to define a "feedable" iterator where you can choose
    between concrete iterators by feeding a value in a `tf.Session.run` call.
    In that case, `string_handle` would be a `tf.compat.v1.placeholder`, and you
    would
    feed it with the value of `tf.data.Iterator.string_handle` in each step.

    For example, if you had two iterators that marked the current position in
    a training dataset and a test dataset, you could choose which to use in
    each step as follows:

    ```python
    train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())

    test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_element = iterator.get_next()
    loss = f(next_element)

    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
    ```

    Args:
      string_handle: A scalar `tf.Tensor` of type `tf.string` that evaluates to
        a handle produced by the `Iterator.string_handle()` method.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset. If
        omitted, each component will have an unconstrainted shape.
      output_classes: (Optional.) A nested structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.
    """
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(output_types,
                                               tensor_shape.as_shape,
                                               output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: ops.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    output_structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    string_handle = ops.convert_to_tensor(string_handle, dtype=dtypes.string)
    if _device_stack_is_empty():
      with ops.device("/cpu:0"):
        iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
            string_handle,
            output_types=structure.get_flat_tensor_types(output_structure),
            output_shapes=structure.get_flat_tensor_shapes(output_structure))
    else:
      iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
          string_handle,
          output_types=structure.get_flat_tensor_types(output_structure),
          output_shapes=structure.get_flat_tensor_shapes(output_structure))
    return Iterator(iterator_resource, None, output_types, output_shapes,
                    output_classes)

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

  def make_initializer(self, dataset, name=None):
    """Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` with compatible structure to this iterator.
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        element structure.
    """
    with ops.name_scope(name, "make_initializer") as name:
      # NOTE(mrry): Cannot depend on `dataset_ops.get_legacy_output*()` due
      # to that creating a circular dependency.
      # pylint: disable=protected-access
      dataset_output_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),
          dataset.element_spec)
      dataset_output_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),
          dataset.element_spec)
      dataset_output_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),
          dataset.element_spec)
      # pylint: enable=protected-access

      nest.assert_same_structure(self.output_types, dataset_output_types)
      nest.assert_same_structure(self.output_shapes, dataset_output_shapes)
      for iterator_class, dataset_class in zip(
          nest.flatten(self.output_classes),
          nest.flatten(dataset_output_classes)):
        if iterator_class is not dataset_class:
          raise TypeError(
              "Expected output classes %r but got dataset with output class %r."
              % (self.output_classes, dataset_output_classes))
      for iterator_dtype, dataset_dtype in zip(
          nest.flatten(self.output_types), nest.flatten(dataset_output_types)):
        if iterator_dtype != dataset_dtype:
          raise TypeError(
              "Expected output types %r but got dataset with output types %r." %
              (self.output_types, dataset_output_types))
      for iterator_shape, dataset_shape in zip(
          nest.flatten(self.output_shapes), nest.flatten(
              dataset_output_shapes)):
        if not iterator_shape.is_compatible_with(dataset_shape):
          raise TypeError("Expected output shapes compatible with %r but got "
                          "dataset with output shapes %r." %
                          (self.output_shapes, dataset_output_shapes))
    with ops.colocate_with(self._iterator_resource):
      return gen_dataset_ops.make_iterator(
          dataset._variant_tensor, self._iterator_resource, name=name)  # pylint: disable=protected-access

  def get_next(self, name=None):
    """Returns a nested structure of `tf.Tensor`s representing the next element.

    In graph mode, you should typically call this method *once* and use its
    result as the input to another computation. A typical loop will then call
    `tf.Session.run` on the result of that computation. The loop will terminate
    when the `Iterator.get_next()` operation raises
    `tf.errors.OutOfRangeError`. The following skeleton shows how to use
    this method when building a training loop:

    ```python
    dataset = ...  # A `tf.data.Dataset` object.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Build a TensorFlow graph that does something with each element.
    loss = model_function(next_element)
    optimizer = ...  # A `tf.compat.v1.train.Optimizer` object.
    train_op = optimizer.minimize(loss)

    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(train_op)
      except tf.errors.OutOfRangeError:
        pass
    ```

    NOTE: It is legitimate to call `Iterator.get_next()` multiple times, e.g.
    when you are distributing different elements to multiple devices in a single
    step. However, a common pitfall arises when users call `Iterator.get_next()`
    in each iteration of their training loop. `Iterator.get_next()` adds ops to
    the graph, and executing each op allocates resources (including threads); as
    a consequence, invoking it in every iteration of a training loop causes
    slowdown and eventual resource exhaustion. To guard against this outcome, we
    log a warning when the number of uses crosses a fixed threshold of
    suspiciousness.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    self._get_next_call_count += 1
    if self._get_next_call_count > GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)

    # pylint: disable=protected-access
    flat_ret = gen_dataset_ops.iterator_get_next(
        self._iterator_resource,
        output_types=self._flat_tensor_types,
        output_shapes=self._flat_tensor_shapes,
        name=name)
    return structure.from_tensor_list(self._element_spec, flat_ret)

  def string_handle(self, name=None):
    """Returns a string-valued `tf.Tensor` that represents this iterator.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.string`.
    """
    if name is None:
      return self._string_handle
    else:
      return gen_dataset_ops.iterator_to_string_handle(
          self._iterator_resource, name=name)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(iterator)`.")
  def output_classes(self):
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(iterator)`.")
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(iterator)`.")
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def element_spec(self):
    """The type specification of an element of this iterator.

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    """
    return self._element_spec

  def _gather_saveables_for_checkpoint(self):

    def _saveable_factory(name):
      return _IteratorSaveable(self._iterator_resource, name)

    return {"ITERATOR": _saveable_factory}


_uid_counter = 0
_uid_lock = threading.Lock()


def _generate_shared_name(prefix):
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return "{}{}".format(prefix, uid)


class IteratorResourceDeleter(object):
  """An object which cleans up an iterator resource handle.

  An alternative to defining a __del__ method on an object. Even if the parent
  object is part of a reference cycle, the cycle will be collectable.
  """

  def __init__(self, handle, device, deleter):
    self._deleter = deleter
    self._handle = handle
    self._device = device
    self._eager_mode = context.executing_eagerly()

  def __del__(self):
    with ops.device(self._device):
      # Make sure the resource is deleted in the same mode as it was created in.
      if self._eager_mode:
        with context.eager_mode():
          gen_dataset_ops.delete_iterator(
              handle=self._handle, deleter=self._deleter)
      else:
        with context.graph_mode():
          gen_dataset_ops.delete_iterator(
              handle=self._handle, deleter=self._deleter)


class OwnedIterator(trackable.Trackable, composite_tensor.CompositeTensor):
  """An iterator producing tf.Tensor objects from a tf.data.Dataset.

  The iterator resource  created through `OwnedIterator` is owned by the Python
  object and the life time of the underlying resource is tied to the life time
  of the `OwnedIterator` object. This makes `OwnedIterator` appropriate for use
  in eager mode and inside of tf.functions.
  """

  def __init__(self, dataset=None, components=None, element_spec=None):
    """Creates a new iterator from the given dataset.

    If `dataset` is not specified, the iterator will be created from the given
    tensor components and element structure. In particular, the alternative for
    constructing the iterator is used when the iterator is reconstructed from
    it `CompositeTensor` representation.

    Args:
      dataset: A `tf.data.Dataset` object.
      components: Tensor components to construct the iterator from.
      element_spec: A nested structure of `TypeSpec` objects that
        represents the type specification of elements of the iterator.

    Raises:
      ValueError: If `dataset` is not provided and either `components` or
        `element_spec` is not provided. Or `dataset` is provided and either
        `components` and `element_spec` is provided.
    """

    error_message = "Either `dataset` or both `components` and "
    "`element_spec` need to be provided."

    self._device = context.context().device_name

    if dataset is None:
      if (components is None or element_spec is None):
        raise ValueError(error_message)
      # pylint: disable=protected-access
      self._element_spec = element_spec
      self._flat_output_types = structure.get_flat_tensor_types(
          self._element_spec)
      self._flat_output_shapes = structure.get_flat_tensor_shapes(
          self._element_spec)
      self._iterator_resource, self._deleter = components
    else:
      if (components is not None or element_spec is not None):
        raise ValueError(error_message)
      if (_device_stack_is_empty() or
          context.context().device_spec.device_type != "CPU"):
        with ops.device("/cpu:0"):
          self._create_iterator(dataset)
      else:
        self._create_iterator(dataset)

  def _create_iterator(self, dataset):
    # pylint: disable=protected-access
    dataset = dataset._apply_options()

    # Store dataset reference to ensure that dataset is alive when this iterator
    # is being used. For example, `tf.data.Dataset.from_generator` registers
    # a few py_funcs that are needed in `self._next_internal`.  If the dataset
    # is deleted, this iterator crashes on `self.__next__(...)` call.
    self._dataset = dataset

    ds_variant = dataset._variant_tensor
    self._element_spec = dataset.element_spec
    self._flat_output_types = structure.get_flat_tensor_types(
        self._element_spec)
    self._flat_output_shapes = structure.get_flat_tensor_shapes(
        self._element_spec)
    with ops.colocate_with(ds_variant):
      self._iterator_resource, self._deleter = (
          gen_dataset_ops.anonymous_iterator_v2(
              output_types=self._flat_output_types,
              output_shapes=self._flat_output_shapes))
      gen_dataset_ops.make_iterator(ds_variant, self._iterator_resource)
      # Delete the resource when this object is deleted
      self._resource_deleter = IteratorResourceDeleter(
          handle=self._iterator_resource,
          device=self._device,
          deleter=self._deleter)

  def __iter__(self):
    return self

  def __next__(self):  # For Python 3 compatibility
    return self.next()

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    if not context.executing_eagerly():
      with ops.device(self._device):
        ret = gen_dataset_ops.iterator_get_next(
            self._iterator_resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)
      return structure.from_compatible_tensor_list(self._element_spec, ret)

    # This runs in sync mode as iterators use an error status to communicate
    # that there is no more data to iterate over.
    # TODO(b/77291417): Fix
    with context.execution_mode(context.SYNC):
      with ops.device(self._device):
        # TODO(ashankar): Consider removing this ops.device() contextmanager
        # and instead mimic ops placement in graphs: Operations on resource
        # handles execute on the same device as where the resource is placed.
        ret = gen_dataset_ops.iterator_get_next(
            self._iterator_resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)

      try:
        # Fast path for the case `self._structure` is not a nested structure.
        return self._element_spec._from_compatible_tensor_list(ret)  # pylint: disable=protected-access
      except AttributeError:
        return structure.from_compatible_tensor_list(self._element_spec, ret)

  @property
  def _type_spec(self):
    return IteratorSpec(self.element_spec)

  def next(self):
    """Returns a nested structure of `Tensor`s containing the next element."""
    try:
      return self._next_internal()
    except errors.OutOfRangeError:
      raise StopIteration

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(iterator)`.")
  def output_classes(self):
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(iterator)`.")
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(iterator)`.")
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def element_spec(self):
    """The type specification of an element of this iterator.

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    """
    return self._element_spec

  def get_next(self, name=None):
    """Returns a nested structure of `tf.Tensor`s containing the next element.

    Args:
      name: (Optional.) A name for the created operation. Currently unused.

    Returns:
      A nested structure of `tf.Tensor` objects.

    Raises:
      `tf.errors.OutOfRangeError`: If the end of the dataset has been reached.
    """
    del name
    return self._next_internal()

  def _gather_saveables_for_checkpoint(self):

    def _saveable_factory(name):
      return _IteratorSaveable(self._iterator_resource, name)

    return {"ITERATOR": _saveable_factory}


# TODO(jsimsa): Export this as "tf.data.IteratorSpec".
class IteratorSpec(type_spec.TypeSpec):
  """Type specification for `OwnedIterator`."""

  __slots__ = ["_element_spec"]

  def __init__(self, element_spec):
    self._element_spec = element_spec

  @property
  def value_type(self):
    return OwnedIterator

  def _serialize(self):
    return (self._element_spec,)

  @property
  def _component_specs(self):
    return (
        tensor_spec.TensorSpec([], dtypes.resource),
        tensor_spec.TensorSpec([], dtypes.variant),
    )

  def _to_components(self, value):
    return (value._iterator_resource, value._deleter)  # pylint: disable=protected-access

  def _from_components(self, components):
    return OwnedIterator(
        dataset=None,
        components=components,
        element_spec=self._element_spec)

  @staticmethod
  def from_value(value):
    return IteratorSpec(value.element_spec)  # pylint: disable=protected-access


# TODO(b/71645805): Expose trackable stateful objects from dataset.
class _IteratorSaveable(BaseSaverBuilder.SaveableObject):
  """SaveableObject for saving/restoring iterator state."""

  def __init__(
      self,
      iterator_resource,
      name,
      external_state_policy=distribute_options.ExternalStatePolicy.FAIL):
    serialized_iterator = gen_dataset_ops.serialize_iterator(
        iterator_resource, external_state_policy=external_state_policy.value)
    specs = [
        BaseSaverBuilder.SaveSpec(
            serialized_iterator,
            "",
            name + "_STATE",
            device=iterator_resource.device)
    ]
    super(_IteratorSaveable, self).__init__(iterator_resource, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    with ops.colocate_with(self.op):
      return gen_dataset_ops.deserialize_iterator(self.op, restored_tensors[0])


@tf_export("data.experimental.get_next_as_optional")
def get_next_as_optional(iterator):
  """Returns an `Optional` that contains the next value from the iterator.

  If `iterator` has reached the end of the sequence, the returned `Optional`
  will have no value.

  Args:
    iterator: A `tf.compat.v1.data.Iterator` object.

  Returns:
    An `Optional` object representing the next value from the iterator (if it
    has one) or no value.
  """
  # pylint: disable=protected-access
  return optional_ops._OptionalImpl(
      gen_dataset_ops.iterator_get_next_as_optional(
          iterator._iterator_resource,
          output_types=structure.get_flat_tensor_types(iterator.element_spec),
          output_shapes=structure.get_flat_tensor_shapes(
              iterator.element_spec)), iterator.element_spec)
