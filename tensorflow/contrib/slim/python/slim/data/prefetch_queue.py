# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements a simple prefetch_queue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner


def prefetch_queue(tensors,
                   capacity=8,
                   shared_name=None,
                   name=None):
  """Creates a queue to prefetech tensors from `tensors`.

  A queue runner for enqueing tensors into the prefetch_queue is automatically
  added to the TF QueueRunners collection.

  Example:
  This is for example useful to pre-assemble input batches read with
  `tf.train.batch()` and enqueue the pre-assembled batches.  Ops that dequeue
  from the pre-assembled queue will not pay the cost of assembling the batch.

  images, labels = tf.train.batch([image, label], batch_size=32, num_threads=4)
  batch_queue = prefetch_queue([images, labels])
  images, labels = batch_queue.dequeue()
  logits = Net(images)
  loss = Loss(logits, labels)

  Args:
    tensors: A list or dictionary of `Tensors` to enqueue in the buffer.
    capacity: An integer. The maximum number of elements in the queue.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A queue from which you can dequeue tensors with the same type and shape
    as `tensors`.
  """
  if isinstance(tensors, dict):
    # Need to wrap the keys and values in list() since Python3 returns views.
    # We sort the keys so the order is consistent across runs.
    names = list(sorted(tensors.keys()))
    tensor_list = list([tensors[n] for n in names])
  else:
    names = None
    tensor_list = tensors

  with ops.name_scope(name, "prefetch_queue", tensor_list) as name:
    dtypes = [t.dtype for t in tensor_list]
    shapes = [t.get_shape() for t in tensor_list]
    queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                    dtypes=dtypes,
                                    shapes=shapes,
                                    names=names,
                                    shared_name=shared_name)
    enqueue_op = queue.enqueue(tensors, name=name)
    queue_runner.add_queue_runner(
        queue_runner.QueueRunner(queue, [enqueue_op]))
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.to_float(queue.size()) * (1. / capacity))
    return queue
