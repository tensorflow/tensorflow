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

import warnings

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops


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
    "`next_element = iterator.get_next() once outside the loop, and use "
    "`next_element` inside the loop.")


class Iterator(object):
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
        each component of an element of this dataset.
      output_shapes: A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset.
      output_classes: A nested structure of Python `type` object corresponding
        to each
        component of an element of this iterator.
    """
    self._iterator_resource = iterator_resource
    self._initializer = initializer
    self._output_classes = output_classes
    self._output_types = output_types
    self._output_shapes = output_shapes
    self._string_handle = gen_dataset_ops.iterator_to_string_handle(
        self._iterator_resource)
    self._get_next_call_count = 0

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
    iterator_resource = gen_dataset_ops.iterator(
        container="",
        shared_name=shared_name,
        output_types=nest.flatten(output_types),
        output_shapes=nest.flatten(output_shapes))
    return Iterator(iterator_resource, None, output_types, output_shapes,
                    output_classes)

  @staticmethod
  def from_string_handle(string_handle,
                         output_types,
                         output_shapes=None,
                         output_classes=None):
    """Creates a new, uninitialized `Iterator` based on the given handle.

    This method allows you to define a "feedable" iterator where you can choose
    between concrete iterators by feeding a value in a @{tf.Session.run} call.
    In that case, `string_handle` would a @{tf.placeholder}, and you would feed
    it with the value of @{tf.data.Iterator.string_handle} in each step.

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
    iterator_resource = gen_dataset_ops.iterator_from_string_handle(
        string_handle,
        output_types=nest.flatten(output_types),
        output_shapes=nest.flatten(output_shapes))
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
      nest.assert_same_structure(self._output_types, dataset.output_types)
      nest.assert_same_structure(self._output_shapes, dataset.output_shapes)
      for iterator_class, dataset_class in zip(
          nest.flatten(self._output_classes),
          nest.flatten(dataset.output_classes)):
        if iterator_class is not dataset_class:
          raise TypeError(
              "Expected output classes %r but got dataset with output class %r."
              % (self._output_classes, dataset.output_classes))
      for iterator_dtype, dataset_dtype in zip(
          nest.flatten(self._output_types), nest.flatten(dataset.output_types)):
        if iterator_dtype != dataset_dtype:
          raise TypeError(
              "Expected output types %r but got dataset with output types %r." %
              (self._output_types, dataset.output_types))
      for iterator_shape, dataset_shape in zip(
          nest.flatten(self._output_shapes), nest.flatten(
              dataset.output_shapes)):
        if not iterator_shape.is_compatible_with(dataset_shape):
          raise TypeError("Expected output shapes compatible with %r but got "
                          "dataset with output shapes %r." %
                          (self._output_shapes, dataset.output_shapes))
    with ops.colocate_with(self._iterator_resource):
      return gen_dataset_ops.make_iterator(
          dataset._as_variant_tensor(), self._iterator_resource, name=name)  # pylint: disable=protected-access

  def get_next(self, name=None):
    """Returns a nested structure of `tf.Tensor`s containing the next element.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    self._get_next_call_count += 1
    if self._get_next_call_count > GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)

    return sparse.deserialize_sparse_tensors(
        nest.pack_sequence_as(self._output_types,
                              gen_dataset_ops.iterator_get_next(
                                  self._iterator_resource,
                                  output_types=nest.flatten(
                                      sparse.as_dense_types(
                                          self._output_types,
                                          self._output_classes)),
                                  output_shapes=nest.flatten(
                                      sparse.as_dense_shapes(
                                          self._output_shapes,
                                          self._output_classes)),
                                  name=name)), self._output_types,
        self._output_shapes, self._output_classes)

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
    return self._output_classes

  @property
  def output_shapes(self):
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return self._output_shapes

  @property
  def output_types(self):
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return self._output_types
