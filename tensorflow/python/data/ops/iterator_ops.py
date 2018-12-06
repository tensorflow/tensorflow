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

from tensorflow.python.compat import compat
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.data.util import structure as structure_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.saver import BaseSaverBuilder
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
  # pylint: disable=protected-access
  device_stack = ops.get_default_graph()._device_functions_outer_to_inner
  # pylint: enable=protected-access
  return not bool(device_stack)


@tf_export(v1=["data.Iterator"])
class Iterator(checkpointable.CheckpointableBase):
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
    self._structure = structure_lib.convert_legacy_structure(
        output_types, output_shapes, output_classes)

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
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: ops.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    if shared_name is None:
      shared_name = ""
    if compat.forward_compatible(2018, 8, 3):
      if _device_stack_is_empty():
        with ops.device("/cpu:0"):
          iterator_resource = gen_dataset_ops.iterator_v2(
              container="",
              shared_name=shared_name,
              output_types=nest.flatten(
                  sparse.as_dense_types(output_types, output_classes)),
              output_shapes=nest.flatten(
                  sparse.as_dense_shapes(output_shapes, output_classes)))
      else:
        iterator_resource = gen_dataset_ops.iterator_v2(
            container="",
            shared_name=shared_name,
            output_types=nest.flatten(
                sparse.as_dense_types(output_types, output_classes)),
            output_shapes=nest.flatten(
                sparse.as_dense_shapes(output_shapes, output_classes)))
    else:
      iterator_resource = gen_dataset_ops.iterator(
          container="",
          shared_name=shared_name,
          output_types=nest.flatten(
              sparse.as_dense_types(output_types, output_classes)),
          output_shapes=nest.flatten(
              sparse.as_dense_shapes(output_shapes, output_classes)))
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
    In that case, `string_handle` would be a `tf.placeholder`, and you would
    feed it with the value of `tf.data.Iterator.string_handle` in each step.

    For example, if you had two iterators that marked the current position in
    a training dataset and a test dataset, you could choose which to use in
    each step as follows:

    ```python
    train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())

    test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_element = iterator.get_next()
    loss = f(next_element)

    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
    ```

    Args:
      string_handle: A scalar `tf.Tensor` of type `tf.string` that evaluates
        to a handle produced by the `Iterator.string_handle()` method.
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
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: ops.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    string_handle = ops.convert_to_tensor(string_handle, dtype=dtypes.string)
    if compat.forward_compatible(2018, 8, 3):
      if _device_stack_is_empty():
        with ops.device("/cpu:0"):
          iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
              string_handle,
              output_types=nest.flatten(
                  sparse.as_dense_types(output_types, output_classes)),
              output_shapes=nest.flatten(
                  sparse.as_dense_shapes(output_shapes, output_classes)))
      else:
        iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
            string_handle,
            output_types=nest.flatten(
                sparse.as_dense_types(output_types, output_classes)),
            output_shapes=nest.flatten(
                sparse.as_dense_shapes(output_shapes, output_classes)))
    else:
      iterator_resource = gen_dataset_ops.iterator_from_string_handle(
          string_handle,
          output_types=nest.flatten(
              sparse.as_dense_types(output_types, output_classes)),
          output_shapes=nest.flatten(
              sparse.as_dense_shapes(output_shapes, output_classes)))
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
      nest.assert_same_structure(self.output_types, dataset.output_types)
      nest.assert_same_structure(self.output_shapes, dataset.output_shapes)
      for iterator_class, dataset_class in zip(
          nest.flatten(self.output_classes),
          nest.flatten(dataset.output_classes)):
        if iterator_class is not dataset_class:
          raise TypeError(
              "Expected output classes %r but got dataset with output class %r."
              % (self.output_classes, dataset.output_classes))
      for iterator_dtype, dataset_dtype in zip(
          nest.flatten(self.output_types), nest.flatten(dataset.output_types)):
        if iterator_dtype != dataset_dtype:
          raise TypeError(
              "Expected output types %r but got dataset with output types %r." %
              (self.output_types, dataset.output_types))
      for iterator_shape, dataset_shape in zip(
          nest.flatten(self.output_shapes), nest.flatten(
              dataset.output_shapes)):
        if not iterator_shape.is_compatible_with(dataset_shape):
          raise TypeError("Expected output shapes compatible with %r but got "
                          "dataset with output shapes %r." %
                          (self.output_shapes, dataset.output_shapes))
    with ops.colocate_with(self._iterator_resource):
      return gen_dataset_ops.make_iterator(
          dataset._as_variant_tensor(), self._iterator_resource, name=name)  # pylint: disable=protected-access

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
    optimizer = ...  # A `tf.train.Optimizer` object.
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
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
        output_types=self._structure._flat_types,
        output_shapes=self._structure._flat_shapes, name=name)
    return self._structure._from_tensor_list(flat_ret)

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
  def output_classes(self):
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return self._structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return self._structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return self._structure._to_legacy_output_types()  # pylint: disable=protected-access

  @property
  def _element_structure(self):
    """The structure of an element of this iterator.

    Returns:
      A `Structure` object representing the structure of the components of this
        optional.
    """
    return self._structure

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


class EagerIterator(checkpointable.CheckpointableBase):
  """An iterator producing tf.Tensor objects from a tf.data.Dataset."""

  def __init__(self, dataset):
    """Creates a new iterator over the given dataset.

    For example:
    ```python
    dataset = tf.data.Dataset.range(4)
    for x in Iterator(dataset):
      print(x)
    ```

    Tensors produced will be placed on the device on which this iterator object
    was created.

    Args:
      dataset: A `tf.data.Dataset` object.

    Raises:
      RuntimeError: When invoked without eager execution enabled.
    """

    if not context.executing_eagerly():
      raise RuntimeError(
          "{} objects can only be used when eager execution is enabled, use "
          "tf.data.Dataset.make_initializable_iterator or "
          "tf.data.Dataset.make_one_shot_iterator for graph construction".
          format(type(self)))
    self._device = context.context().device_name
    with ops.device("/cpu:0"):
      # pylint: disable=protected-access
      ds_variant = dataset._as_variant_tensor()
      self._structure = structure_lib.convert_legacy_structure(
          dataset.output_types, dataset.output_shapes, dataset.output_classes)
      self._flat_output_types = self._structure._flat_types
      self._flat_output_shapes = self._structure._flat_shapes
      with ops.colocate_with(ds_variant):
        self._resource = gen_dataset_ops.anonymous_iterator(
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)
        gen_dataset_ops.make_iterator(ds_variant, self._resource)
        # Delete the resource when this object is deleted
        self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
            handle=self._resource, handle_device=self._device)

  def __iter__(self):
    return self

  def __next__(self):  # For Python 3 compatibility
    return self.next()

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    # This runs in sync mode as iterators use an error status to communicate
    # that there is no more data to iterate over.
    # TODO(b/77291417): Fix
    with context.execution_mode(context.SYNC):
      with ops.device(self._device):
        # TODO(ashankar): Consider removing this ops.device() contextmanager
        # and instead mimic ops placement in graphs: Operations on resource
        # handles execute on the same device as where the resource is placed.
        # NOTE(mrry): Here we use the "_sync" variant of `iterator_get_next`
        # because in eager mode this code will run synchronously on the calling
        # thread. Therefore we do not need to make a defensive context switch
        # to a background thread, and can achieve a small constant performance
        # boost by invoking the iterator synchronously.
        ret = gen_dataset_ops.iterator_get_next_sync(
            self._resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)

      return self._structure._from_compatible_tensor_list(ret)  # pylint: disable=protected-access

  def next(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    try:
      return self._next_internal()
    except errors.OutOfRangeError:
      raise StopIteration

  @property
  def output_classes(self):
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return self._structure._to_legacy_output_classes()  # pylint: disable=protected-access

  @property
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return self._structure._to_legacy_output_shapes()  # pylint: disable=protected-access

  @property
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return self._structure._to_legacy_output_types()  # pylint: disable=protected-access

  @property
  def _element_structure(self):
    """The structure of an element of this iterator.

    Returns:
      A `Structure` object representing the structure of the components of this
        optional.
    """
    return self._structure

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
      return _IteratorSaveable(self._resource, name)

    return {"ITERATOR": _saveable_factory}


# TODO(b/71645805): Expose checkpointable stateful objects from dataset
# attributes(potential).
class _IteratorSaveable(BaseSaverBuilder.SaveableObject):
  """SaveableObject for saving/restoring iterator state."""

  def __init__(self, iterator_resource, name):
    serialized_iterator = gen_dataset_ops.serialize_iterator(iterator_resource)
    specs = [
        BaseSaverBuilder.SaveSpec(serialized_iterator, "", name + "_STATE")
    ]
    super(_IteratorSaveable, self).__init__(iterator_resource, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    with ops.colocate_with(self.op):
      return gen_dataset_ops.deserialize_iterator(self.op, restored_tensors[0])


def get_next_as_optional(iterator):
  """Returns an `Optional` that contains the next value from the iterator.

  If `iterator` has reached the end of the sequence, the returned `Optional`
  will have no value.

  Args:
    iterator: A `tf.data.Iterator` object.

  Returns:
    An `Optional` object representing the next value from the iterator (if it
    has one) or no value.
  """
  # pylint: disable=protected-access
  return optional_ops._OptionalImpl(
      gen_dataset_ops.iterator_get_next_as_optional(
          iterator._iterator_resource,
          output_types=iterator._element_structure._flat_types,
          output_shapes=iterator._element_structure._flat_shapes),
      iterator._element_structure)
