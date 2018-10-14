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

"""Adadelta for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import adadelta
from tensorflow.python.util import deprecation


class AdadeltaOptimizer(adadelta.Adadelta):
  """Optimizer that implements the Adadelta algorithm.

  See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
  ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))
  """

  @deprecation.deprecated_args(
      "2018-10-01",
      "`use_locking = True` is no longer supported and will be ignored.",
      ("use_locking", [False]))
  def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-8,
               use_locking=False, name="Adadelta"):
    """Construct a new Adadelta optimizer.

    Some of the args below are hyperparameters, where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
        To match the exact form in the original paper use 1.0.
      rho: A float hyperparameter. The decay rate.
      epsilon: A float hyperparameter. A constant epsilon used to better
        condition the grad update.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".
    """
    super(AdadeltaOptimizer, self).__init__(
        learning_rate=learning_rate, rho=rho, epsilon=epsilon, name=name)
