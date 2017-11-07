# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""SGDR learning rate decay function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops


def sgdr_decay(learning_rate, global_step, initial_period_steps,
               t_mul=2.0, m_mul=1.0, name=None):
  """Implements Stochastic Gradient Descent with Warm Restarts (SGDR).
  
  As described in "SGDR: Stochastic Gradient Descent
  with Warm Restarts" by Ilya Loshchilov & Frank Hutter, Proceedings of
  ICLR'2017, available at https://arxiv.org/pdf/1608.03983.pdf

  The learning rate decreases according to cosine annealing:

  ```python
  learning_rate * 0.5 * (1 + cos(x_val * pi)) # for x_val defined in [0, 1]
  ```

  Thus, at the beginning (when the restart index i = 0),
  the learning rate decreases for `initial_period_steps` steps from the initial
  learning rate `learning_rate` (when `x_val=0`, we get `cos(0)=1`) to
  0 (when `x_val=1`, we get `cos(pi)=-1`).

  The decrease within the i-th period takes `t_i` steps,
  where `t_0` = `initial_period_steps` is the user-defined number of batch
  iterations (not epochs as in the paper) to be performed before the first
  restart is launched.
  
  Then, we perform the first restart (i=1) by setting the learning rate to
  `learning_rate*(m_mul^i)`, where `m_mul in [0,1]` (set to 1 by default).
  The i-th restart runs for `t_i=t_0*(t_mul^i)` steps, i.e., every new
  restart runs `t_mul` times longer than the previous one.

  Importantly, when one has no access to a validation set, SGDR suggests
  to report the best expected / recommended solution in the following way:
  When we are within our initial run (i=0), every new solution represents
  SGDR's recommended solution. Instead, when i>0, the recommended solution is
  the one obtained at the end of each restart.

  Note that the minimum learning rate is set to 0 for simplicity,
  you can adjust the code to deal with any positive minimum learning rate
  as defined in the paper.

  `initial_period_steps` is the duration of the first period measured in terms
  of number of minibatch updates. If one wants to use epochs, one should compute
  the number of updates required for an epoch.

  For example, assume the following parameters and intention:
      Minibatch size: 100
      Training dataset size: 10000
      If the user wants the first decay period to span across 5 epochs, then
      `initial_period_steps` = 5 * 10000/100 = 500
  
      Train for 10000 batch iterations with the initial learning rate set to
      0.1, then restart to run 2 times longer, i.e, for 20000 batch iterations
      and with the initial learning rate 0.05, then restart again and again,
      doubling the runtime of each new period and with two times smaller
      initial learning rate.

  To accomplish the above, one would write:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = sgdr_decay(starter_learning_rate, global_step,
                             initial_period_steps=10000, t_mul=2, m_mul=0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )

  # Step  | 0   | 1000  | 5000 | 9000  | 9999 | 10000 | 11000  |
  # LR    | 0.1 | 0.097 | 0.05 | 0.002 | 0.00 | 0.05  | 0.0496 |

  # Step  | 20000 | 29000  | 29999 | 30000 |
  # LR    | 0.025 | 0.0003 | 0.00  | 0.025 |
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    initial_period_steps: Duration of the first period measured as the number
      of minibatch updates, if one wants to use epochs, one should compute
      the number of updates required for an epoch.
    t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
      Must be positive.
      Used to derive the number of iterations in the i-th period:
      `initial_period_steps * (t_mul^i)`. Defaults to 2.0.
    m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
      Must be positive.
      Used to derive the initial learning rate of the i-th period:
      `learning_rate * (m_mul^i)`. Defaults to 1.0

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.
    The learning rate for a provided global_step.
  Raises:
    ValueError: if `global_step` is not supplied.
  """

  if global_step is None:
    raise ValueError("global_step is required for sgdr_decay.")
  with ops.name_scope(name, "SGDRDecay",
                      [learning_rate, global_step,
                       initial_period_steps, t_mul, m_mul]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate,
                                          name="initial_learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    t_0 = math_ops.cast(initial_period_steps, dtype)
    t_mul = math_ops.cast(t_mul, dtype)
    m_mul = math_ops.cast(m_mul, dtype)

    c_one = math_ops.cast(constant_op.constant(1.0), dtype)
    c_half = math_ops.cast(constant_op.constant(0.5), dtype)
    c_pi = math_ops.cast(constant_op.constant(math.pi), dtype)

    # Find normalized value of the current step
    x_val = math_ops.div(global_step, t_0)

    def compute_step(x_val, geometric=False):
      if geometric:
        # Consider geometric series where t_mul != 1
        # 1 + t_mul + t_mul^2 ... = (1 - t_mul^i_restart) / (1 - t_mul)

        # First find how many restarts were performed for a given x_val
        # Find maximal integer i_restart value for which this equation holds
        # x_val >= (1 - t_mul^i_restart) / (1 - t_mul)
        # x_val * (1 - t_mul) <= (1 - t_mul^i_restart)
        # t_mul^i_restart <= (1 - x_val * (1 - t_mul))

        # tensorflow allows only log with base e
        # i_restart <= log(1 - x_val * (1 - t_mul) / log(t_mul)
        # Find how many restarts were performed

        i_restart = math_ops.floor(
            math_ops.log(c_one - x_val * (c_one - t_mul)) / math_ops.log(t_mul))
        # Compute the sum of all restarts before the current one
        sum_r = (c_one - t_mul ** i_restart) / (c_one - t_mul)
        # Compute our position within the current restart
        x_val = (x_val - sum_r) / t_mul ** i_restart

      else:
        # Find how many restarts were performed
        i_restart = math_ops.floor(x_val)
        # Compute our position within the current restart
        x_val = x_val - i_restart
      return i_restart, x_val

    i_restart, x_val = control_flow_ops.cond(
        math_ops.equal(t_mul, c_one),
        lambda: compute_step(x_val, geometric=False),
        lambda: compute_step(x_val, geometric=True))

    # If m_mul < 1, then the initial learning rate of every new restart will be
    # smaller, i.e., by a factor of m_mul ** i_restart at i_restart-th restart
    m_fac = learning_rate * (m_mul ** i_restart)

  return math_ops.multiply(c_half * m_fac,
                           (math_ops.cos(x_val * c_pi) + c_one), name=name)
