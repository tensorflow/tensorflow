# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Various learning rate decay functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import constant_op


def exponential_time_decay(initial_lr, epoch, decay_rate, name=None):
  """Applies exponential time decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires an `epoch` value to compute
  the decayed learning rate.  You can just pass a TensorFlow variable that you
  increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = initial_lr * exp(-decay_rate * epoch)
  ```

  Example: decay exponetially with a base of 0.96:

  ```python
  ...
  epoch = tf.Variable(0, trainable=False)
  initial_lr = 0.1
  k = 0.5
  learning_rate = tf.train.exponential_time_decay(initial_lr, epoch, k)

  # Passing epoch to minimize() will increment it at each step.
  learning_step = (
      tf.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=epoch)
  )
  ```

  Args:
    initial_lr: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    epoch: A Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_rate: A Python number.  The decay rate.
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialTimeDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """
  with ops.op_scope([initial_lr, epoch, decay_rate],
                    name, "ExponentialTimeDecay") as name:
    initial_lr = ops.convert_to_tensor(initial_lr, name="learning_rate")
    decay_rate = math_ops.cast(decay_rate, initial_lr.dtype)
    exponent = math_ops.exp(math_ops.mul(math_ops.neg(decay_rate), epoch))
    return math_ops.mul(initial_lr, exponent, name=name)


def inverse_time_decay(initial_lr, epoch, decay_rate, name=None):
  """Applies inverse time decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an inverse decay function
  to a provided initial learning rate.  It requires an `epoch` value to compute
  the decayed learning rate.  You can just pass a TensorFlow variable that you
  increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * t)
  ```

  Example: decay 1/t with a rate of 0.5:

  ```python
  ...
  epoch = tf.Variable(0, trainable=False)
  initial_lr = 0.1
  k = 0.5
  learning_rate = tf.train.inverse_time_decay(initial_lr, epoch, k)

  # Passing epoch to minimize() will increment it at each step.
  learning_step = (
      tf.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=epoch)
  )
  ```

  Args:
    initial_lr: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    epoch: A Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_rate: A Python number.  The decay rate.
    name: String.  Optional name of the operation.  Defaults to
      'InverseTimeDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """

  with ops.op_scope([initial_lr, epoch, decay_rate],
                    name, "InverseTimeDecay") as name:
    initial_lr = ops.convert_to_tensor(initial_lr, name="learning_rate")
    decay_rate = math_ops.cast(decay_rate, initial_lr.dtype)
    const = math_ops.cast(constant_op.constant(1), initial_lr.dtype)
    denom = math_ops.add(const, math_ops.mul(decay_rate, epoch))
    return math_ops.div(initial_lr, denom, name=name)


def step_time_decay(initial_lr, epoch, num_steps_for_decay, decay_rate=0.5,
               name=None):
  """Applies step decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a step decay function
  to a provided initial learning rate.  It requires an `epoch` value to
  compute the decayed learning rate.  You can just pass a python number
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
    count = epoch // num_steps_for_decay
    learning_rate = initial_lr * decay_rate ** count
      
  ```
  ```

  Example: decay every 1000 steps with a base of 0.5:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.step_time_decay(starter_learning_rate, global_step,
                                             1000, 0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    initial_lr: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A Python number.
      Global step to use for the decay computation.  Must not be negative.
    num_steps_for_decay: A Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    name: String.  Optional name of the operation.  Defaults to 'StepTimeDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """
  with ops.op_scope([initial_lr, epoch, num_steps_for_decay, decay_rate],
                    name, "StepTimeDecay") as name:
    initial_lr = ops.convert_to_tensor(initial_lr, name="learning_rate")
    dtype = initial_lr.dtype
    decay_rate = math_ops.cast(decay_rate, dtype)
    count = math_ops.cast(math_ops.div(epoch, num_steps_for_decay),
                          dtypes.int32)
    count = math_ops.cast(count, initial_lr.dtype)
    return math_ops.mul(initial_lr, math_ops.pow(decay_rate, count))
