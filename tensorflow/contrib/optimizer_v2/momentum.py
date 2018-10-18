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

"""Momentum for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import sgd
from tensorflow.python.util import deprecation


class MomentumOptimizer(sgd.SGD):
  """Optimizer that implements the Momentum algorithm.

  Computes (if `use_nesterov = False`):

  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.
  """

  @deprecation.deprecated_args(
      "2018-10-01",
      "`use_locking = True` is no longer supported and will be ignored.",
      ("use_locking", [False]))
  def __init__(self, learning_rate, momentum,
               use_locking=False, name="Momentum", use_nesterov=False):
    """Construct a new Momentum optimizer.

    Some of the args below are hyperparameters, where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
      momentum: A float hyperparameter. The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
        This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.

    @compatibility(eager)
    When eager execution is enabled, learning_rate and momentum can each be a
    callable that takes no arguments and returns the actual value to use. This
    can be useful for changing these values across different invocations of
    optimizer functions.
    @end_compatibility
    """
    super(MomentumOptimizer, self).__init__(
        learning_rate=learning_rate,
        momentum=momentum,
        name=name,
        nesterov=use_nesterov)
