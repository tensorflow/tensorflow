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

"""GradientDescent optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import sgd
from tensorflow.python.util import deprecation


class GradientDescentOptimizer(sgd.SGD):
  """Optimizer that implements the gradient descent algorithm."""

  @deprecation.deprecated_args(
      "2018-10-01",
      "`use_locking = True` is no longer supported and will be ignored.",
      ("use_locking", [False]))
  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.

    The learning rate arg below is a hyperparameter where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(GradientDescentOptimizer, self).__init__(
        learning_rate=learning_rate, name=name)
