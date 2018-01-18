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
"""Iteration over tf.data.Datasets when eager execution is enabled."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops

_uid_counter = 0
_uid_lock = threading.Lock()


def _generate_shared_name(prefix):
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return "{}_{}".format(prefix, uid)


class Iterator(object):
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

    if not context.in_eager_mode():
      raise RuntimeError(
          "{} objects can only be used when eager execution is enabled, use "
          "tf.data.Dataset.make_iterator or "
          "tf.data.Dataset.make_one_shot_iterator for graph construction".
          format(type(self)))
    with ops.device("/device:CPU:0"):
      ds_variant = dataset._as_variant_tensor()  # pylint: disable=protected-access
      self._output_classes = dataset.output_classes
      self._output_types = dataset.output_types
      self._output_shapes = dataset.output_shapes
      self._flat_output_types = nest.flatten(
          sparse.as_dense_types(self._output_types, self._output_classes))
      self._flat_output_shapes = nest.flatten(
          sparse.as_dense_shapes(self._output_shapes, self._output_classes))
      self._resource = gen_dataset_ops.iterator(
          container="",
          shared_name=_generate_shared_name("eager_iterator"),
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)
      gen_dataset_ops.make_iterator(ds_variant, self._resource)
      # Delete the resource when this object is deleted
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device="/device:CPU:0")
    self._device = context.context().device_name
    self._buffer_resource_handle = None
    if not context.context().device_spec.device_type:
      is_remote_device = False
    else:
      is_remote_device = context.context().device_spec.device_type != "CPU"
    if is_remote_device:
      with ops.device("/device:CPU:0"):
        iter_string_handle = gen_dataset_ops.iterator_to_string_handle(
            self._resource)

        @function.Defun(dtypes.string)
        def remote_fn(h):
          remote_iterator = iterator_ops.Iterator.from_string_handle(
              h, self._output_types, self._output_shapes)
          return remote_iterator.get_next()

        remote_fn.add_to_graph(None)
        target = constant_op.constant("/device:CPU:0")
      with ops.device(self._device):
        self._buffer_resource_handle = prefetching_ops.function_buffering_resource(
            string_arg=iter_string_handle,
            f=remote_fn,
            target_device=target,
            buffer_size=10,
            thread_pool_size=1,
            container="",
            shared_name=_generate_shared_name("function_buffer_resource"))
        self._buffer_resource_deleter = resource_variable_ops.EagerResourceDeleter(
            handle=self._buffer_resource_handle, handle_device=self._device)

  def __iter__(self):
    return self

  def __next__(self):  # For Python 3 compatibility
    return self.next()

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    with ops.device(self._device):
      if self._buffer_resource_handle is not None:
        ret = prefetching_ops.function_buffering_resource_get_next(
            function_buffer_resource=self._buffer_resource_handle,
            output_types=self._flat_output_types)
      else:
        # TODO(ashankar): Consider removing this ops.device() contextmanager
        # and instead mimic ops placement in graphs: Operations on resource
        # handles execute on the same device as where the resource is placed.
        ret = gen_dataset_ops.iterator_get_next(
            self._resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)

    return sparse.deserialize_sparse_tensors(
        nest.pack_sequence_as(self._output_types, ret), self._output_types,
        self._output_shapes, self._output_classes)

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
