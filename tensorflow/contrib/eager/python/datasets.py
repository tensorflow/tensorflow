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

from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops

_uid_counter = 0
_uid_lock = threading.Lock()


def _iterator_shared_name():
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return "eager_iterator_{}".format(uid)


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
      self._output_types = dataset.output_types
      self._flat_output_types = nest.flatten(dataset.output_types)
      self._flat_output_shapes = nest.flatten(dataset.output_shapes)
      self._resource = gen_dataset_ops.iterator(
          container="",
          shared_name=_iterator_shared_name(),
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)
      gen_dataset_ops.make_iterator(ds_variant, self._resource)
    self._device = context.context().device_name

  def __del__(self):
    if self._resource is not None:
      with ops.device("/device:CPU:0"):
        resource_variable_ops.destroy_resource_op(self._resource)
    self._resource = None

  def __iter__(self):
    return self

  def __next__(self):  # For Python 3 compatibility
    return self.next()

  def next(self):
    """Return the next tf.Tensor from the dataset."""
    try:
      # TODO(ashankar): Consider removing this ops.device() contextmanager
      # and instead mimic ops placement in graphs: Operations on resource
      # handles execute on the same device as where the resource is placed.
      with ops.device("/device:CPU:0"):
        ret = gen_dataset_ops.iterator_get_next(
            self._resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)
    except errors.OutOfRangeError:
      raise StopIteration
    # Copies tensors from CPU to the current device if necessary.
    # TODO(rohanj): This should be replaced by the mechanism to have the
    # runtime's threads copy tensors to the destination device.
    with ops.device(self._device):
      ret = [array_ops.identity(x) for x in ret]
      return nest.pack_sequence_as(self._output_types, ret)
