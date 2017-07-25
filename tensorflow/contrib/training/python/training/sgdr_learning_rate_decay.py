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


def sgdr_decay(learning_rate, global_step, t_0=1000, t_mul=1, m_mul=1,
               name=None):
  """ This procedure implements Stochastic Gradient Descent with Warm
  Restarts (SGDR) as described in "SGDR: Stochastic Gradient Descent
  with Warm Restarts" by Ilya Loshchilov & Frank Hutter, Proceedings of
  ICLR'2017, available at https://arxiv.org/pdf/1608.03983.pdf
  The basic idea of the algorithm is the following.
  The learning rate decreases according to cosine annealing:

  ```python
  learning_rate * 0.5 * (1 + cos(x_val * pi)) # for x_val defined in [0, 1]
  ```

  Thus, in the initial run (when the restart index i = 0),
  the learning rate decreases from the initial learning rate
  `learning_rate` (when `x_val=0`, we get `cos(0)=1`) to
  0 (when `x_val=1`, we get `cos(pi)=-1`).
  The decrease within i-th restart takes `t_i` steps,
  while `t_0` is user-defined.
  Then, we perform the first restart (i=1) by setting the learning rate to
  `learning_rate*(m_mul^i)`, where `m_mul in [0,1]` (set to 1 by default).
  Also, every restart runs for `t_i=t_0*(t_mul^i)` steps, i.e., every new
  restart runs for `t_mul` longer than the previous one.

  Importantly, when one has no access to a validation set, SGDR suggests
  to report the best expected / recommended solution in the following way.
  When we are within our initial run (i=0), every new solution represents
  SGDR's recommended solution. Instead, when i>0, the recommended solution is
  the one obtained at the end of each restart,
  i.e., when the learning rate is 0.

  Note that the minimum learning rate is set to 0 for simplicity,
  you can adjust the code to deal with any positive minimum learning rate
  as defined in the paper.

  ```
  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    t_0: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  Number of iterations in the first restart,
      it can be set to a multiplicative of the number of batches per epoch.
      Defaults to 1
    t_mul: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.
      Used to derive the number of iterations in the i-th restart:
      `t_0 * (t_mul^i)`. Defaults to 1
    m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
      Must be positive.
      Used to derive the initial learning of the i-th restart:
      `learning_rate * (m_mul^i)`. Defaults to 1
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
                       t_0, t_mul, m_mul]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    t_0 = math_ops.cast(t_0, dtype)
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
