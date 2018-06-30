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
"""Utilities to clip weights.

This is useful in the original formulation of the Wasserstein loss, which
requires that the discriminator be K-Lipschitz. See
https://arxiv.org/pdf/1701.07875 for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.opt.python.training import variable_clipping_optimizer


__all__ = [
    'clip_variables',
    'clip_discriminator_weights',
]


def clip_discriminator_weights(optimizer, model, weight_clip):
  """Modifies an optimizer so it clips weights to a certain value.

  Args:
    optimizer: An optimizer to perform variable weight clipping.
    model: A GANModel namedtuple.
    weight_clip: Positive python float to clip discriminator weights. Used to
      enforce a K-lipschitz condition, which is useful for some GAN training
      schemes (ex WGAN: https://arxiv.org/pdf/1701.07875).

  Returns:
    An optimizer to perform weight clipping after updates.

  Raises:
    ValueError: If `weight_clip` is less than 0.
  """
  return clip_variables(optimizer, model.discriminator_variables, weight_clip)


def clip_variables(optimizer, variables, weight_clip):
  """Modifies an optimizer so it clips weights to a certain value.

  Args:
    optimizer: An optimizer to perform variable weight clipping.
    variables: A list of TensorFlow variables.
    weight_clip: Positive python float to clip discriminator weights. Used to
      enforce a K-lipschitz condition, which is useful for some GAN training
      schemes (ex WGAN: https://arxiv.org/pdf/1701.07875).

  Returns:
    An optimizer to perform weight clipping after updates.

  Raises:
    ValueError: If `weight_clip` is less than 0.
  """
  if weight_clip < 0:
    raise ValueError(
        '`discriminator_weight_clip` must be positive. Instead, was %s',
        weight_clip)
  return variable_clipping_optimizer.VariableClippingOptimizer(
      opt=optimizer,
      # Do no reduction, so clipping happens per-value.
      vars_to_clip_dims={var: [] for var in variables},
      max_norm=weight_clip,
      use_locking=True,
      colocate_clip_ops_with_vars=True)
