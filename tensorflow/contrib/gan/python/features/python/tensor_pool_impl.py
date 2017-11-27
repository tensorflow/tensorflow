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
"""A tensor pool stores values from an input tensor and returns a stored one.

We use this to keep a history of values created by a generator, such that
a discriminator can randomly be trained on some older samples, not just the
current one. This can help to not let the discriminator get too far ahead of the
generator and also to keep the system from oscilating, if the discriminator
forgets too fast what past samples from the generator looked like.

See the following papers for more details.
1) `Learning from simulated and unsupervised images through adversarial
    training` (https://arxiv.org/abs/1612.07828).
2) `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks` (https://arxiv.org/abs/1703.10593).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import random_ops

__all__ = [
    'tensor_pool',
]


def tensor_pool(input_value,
                pool_size,
                pooling_probability=0.5,
                name='tensor_pool'):
  """Queue storing input values and returning random previously stored ones.

  Every time the returned `output_value` is evaluated, `input_value` is
  evaluated and its value either directly returned (with
  `1-pooling_probability`) or stored in the pool and a random one of the samples
  currently in the pool is popped and returned. As long as the pool in not fully
  filled, the input_value is always directly returned, as well as stored in the
  pool. Note during inference / testing, it may be appropriate to set
  `pool_size` = 0 or `pooling_probability` = 0.

  Args:
    input_value: A `Tensor` from which to read values to be pooled.
    pool_size: An integer specifying the maximum size of the pool.
    pooling_probability: A float `Tensor` specifying the probability of getting
      a value from the pool, as opposed to just the current input.
    name: A string prefix for the name scope for all tensorflow ops.

  Returns:
    A `Tensor` which is with given probability either the `input_value` or a
    randomly chosen sample that was previously inserted in the pool.

  Raises:
    ValueError: If `pool_size` is negative.
  """
  pool_size = int(pool_size)
  if pool_size < 0:
    raise ValueError('`pool_size` is negative.')
  elif pool_size == 0:
    return input_value

  with ops.name_scope('{}_pool_queue'.format(name),
                      values=[input_value, pooling_probability]):
    pool_queue = data_flow_ops.RandomShuffleQueue(
        capacity=pool_size,
        min_after_dequeue=0,
        dtypes=[input_value.dtype],
        shapes=None)

    # In pseudeo code this code does the following:
    # if not pool_full:
    #   enqueue(input_value)
    #   return input_value
    # else
    #   dequeue_value = dequeue_random_sample()
    #   enqueue(input_value)
    #   if rand() < pooling_probability:
    #     return dequeue_value
    #   else
    #     return input_value

    def _get_input_value_pooled():
      enqueue_op = pool_queue.enqueue(input_value)
      with ops.control_dependencies([enqueue_op]):
        return array_ops.identity(input_value)

    def _get_random_pool_value_and_enqueue_input():
      dequeue_value = pool_queue.dequeue()
      with ops.control_dependencies([dequeue_value]):
        enqueue_op = pool_queue.enqueue(input_value)
        with ops.control_dependencies([enqueue_op]):
          prob = random_ops.random_uniform(
              (), dtype=dtypes.float32) < pooling_probability
          return control_flow_ops.cond(prob, lambda: dequeue_value,
                                       lambda: input_value)

    output_value = control_flow_ops.cond(
        pool_queue.size() < pool_size, _get_input_value_pooled,
        _get_random_pool_value_and_enqueue_input)

  return output_value
