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
  return "{}{}".format(prefix, uid)


class Iterator(iterator_ops.EagerIterator):
  """An iterator producing tf.Tensor objects from a tf.data.Dataset.

  NOTE: Unlike the iterator created by the
  @{tf.data.Dataset.make_one_shot_iterator} method, this class enables
  additional experimental functionality, such as prefetching to the GPU.
  """

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
    super(Iterator, self).__init__(dataset)
    if not context.context().device_spec.device_type:
      is_remote_device = False
    else:
      is_remote_device = context.context().device_spec.device_type != "CPU"
    self._buffer_resource_handle = None
    if is_remote_device:
      with ops.device("/device:CPU:0"):
        iter_string_handle = gen_dataset_ops.iterator_to_string_handle(
            self._resource)

        @function.Defun(dtypes.string)
        def remote_fn(h):
          remote_iterator = iterator_ops.Iterator.from_string_handle(
              h, self.output_types, self.output_shapes, self.output_classes)
          return remote_iterator.get_next()

        remote_fn.add_to_graph(None)
        target = constant_op.constant("/device:CPU:0")
      with ops.device(self._device):
        self._buffer_resource_handle = prefetching_ops.function_buffering_resource(  # pylint: disable=line-too-long
            string_arg=iter_string_handle,
            f=remote_fn,
            target_device=target,
            buffer_size=10,
            thread_pool_size=1,
            container="",
            shared_name=_generate_shared_name("function_buffer_resource"))
        self._buffer_resource_deleter = resource_variable_ops.EagerResourceDeleter(  # pylint: disable=line-too-long
            handle=self._buffer_resource_handle,
            handle_device=self._device)

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    if self._buffer_resource_handle is not None:
      with ops.device(self._device):
        ret = prefetching_ops.function_buffering_resource_get_next(
            function_buffer_resource=self._buffer_resource_handle,
            output_types=self._flat_output_types)
      return sparse.deserialize_sparse_tensors(
          nest.pack_sequence_as(self._output_types, ret), self._output_types,
          self._output_shapes, self._output_classes)
    else:
      return super(Iterator, self)._next_internal()
