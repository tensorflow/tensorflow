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

"""Adam optimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.util import deprecation


class AdamOptimizer(adam.Adam):
  """Optimizer that implements the Adam algorithm.

  See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
  """

  @deprecation.deprecated_args(
      "2018-10-01",
      "`use_locking = True` is no longer supported and will be ignored.",
      ("use_locking", [False]))
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):
    """Construct a new Adam optimizer.

    Initialization:

    $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
    $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
    $$t := 0 \text{(Initialize timestep)}$$
    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    $$t := t + 1$$
    $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$

    $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
    $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
    $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).

    Some of the args below are hyperparameters where a hyperparameter is
    defined as a scalar Tensor, a regular Python value or a callable (which
    will be evaluated when `apply_gradients` is called) returning a scalar
    Tensor or a Python value.

    Args:
      learning_rate: A float hyperparameter. The learning rate.
      beta1: A float hyperparameter. The exponential decay rate for the 1st
        moment estimates.
      beta2: A float hyperparameter. The exponential decay rate for the 2nd
        moment estimates.
      epsilon: A float hyperparameter. This epsilon is "epsilon hat" in the
        Kingma and Ba paper (in the formula just before Section 2.1), not the
        epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
    """
    super(AdamOptimizer, self).__init__(
        learning_rate=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        name=name)
