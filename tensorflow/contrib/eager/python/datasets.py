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
"""Support for tf.contrib.data when eager execution is enabled."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
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
  """An iterator producing tf.Tensor objects from a tf.contrib.data.Dataset."""

  def __init__(self, dataset):
    """Creates a new iterator over the given dataset.

    For example:
    ```python
    dataset = tf.contrib.data.Dataset.range(4)
    for x in Iterator(dataset):
      print(x)
    ```

    Args:
      dataset: A `tf.contrib.data.Dataset` object.

    Raises:
      RuntimeError: When invoked without eager execution enabled.
    """

    if not context.in_eager_mode():
      raise RuntimeError(
          "{} objects only make sense when eager execution is enabled".format(
              type(self)))
    ds_variant = dataset.make_dataset_resource()
    self._output_types = dataset.output_types
    self._flat_output_types = nest.flatten(dataset.output_types)
    self._flat_output_shapes = nest.flatten(dataset.output_shapes)
    self._resource = gen_dataset_ops.iterator(
        container="",
        shared_name=_iterator_shared_name(),
        output_types=self._flat_output_types,
        output_shapes=self._flat_output_shapes)
    gen_dataset_ops.make_iterator(ds_variant, self._resource)

  def __del__(self):
    if self._resource is not None:
      resource_variable_ops.destroy_resource_op(self._resource)
    self._resource = None

  def __iter__(self):
    return self

  def __next__(self):  # For Python 3 compatibility
    return self.next()

  def next(self):
    """Return the next tf.Tensor from the dataset."""
    try:
      ret = gen_dataset_ops.iterator_get_next(
          self._resource,
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)
      return nest.pack_sequence_as(self._output_types, ret)
    except errors.OutOfRangeError:
      raise StopIteration
