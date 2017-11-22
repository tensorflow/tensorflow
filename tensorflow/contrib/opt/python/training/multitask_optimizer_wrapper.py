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

"""An optimizer wrapper that ensures correct behaviour
of stateful optimizers with multitask loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer

__all__ = ["MultitaskOptimizerWrapper",
           "clip_gradients_by_global_norm"]

def _is_all_zeros(grad):
  all_zeros = math_ops.equal(math_ops.count_nonzero(grad), 0)
  return all_zeros

def _get_wrapper(fn, opt):
  def wrapper(self, grad, *args, **kwargs):  # pylint: disable=unused-argument
    all_zeros = _is_all_zeros(grad)
    return control_flow_ops.cond(
        all_zeros,
        control_flow_ops.no_op,
        lambda: fn(grad, *args, **kwargs))
  wrapper = types.MethodType(wrapper, opt)
  return wrapper

class MultitaskOptimizerWrapper(object):
  """Optimizer wrapper that ensures that
  all-zero gradients don't affect the optimizer state.

  This might be useful when a multi-task loss is used,
  and some components of the loss might be
  not present (e.g. masked out) in some training batches.
  Technically their gradient would be zero,
  which would normally affect the optimizer state
  (e.g. push running average to zero).
  However this is not the desired behaviour,
  since the missing loss component
  should be treated as unknown rather than zero.

  This wrapper filters out all-zero gradient tensors,
  therefore preserving the optimizer state.

  If gradient clipping by global norm is used,
  the provided function clip_gradients_by_global_norm
  should be used (and specified explicitly by the user).
  Otherwise the global norm would be underestimated
  because of all-zero tensors that should be ignored.

  The gradient calculation and application
  are delegated to an underlying optimizer.
  The gradient application is altered only for all-zero tensors.

  Example:
  ```python
  momentum_optimizer = tf.train.MomentumOptimizer(
    learning_rate, momentum=0.9)
  multitask_momentum_optimizer = tf.contrib.opt.MultitaskOptimizerWrapper(
    momentum_optimizer)
  gradvars = multitask_momentum_optimizer.compute_gradients(
    loss)
  gradvars_clipped, _ = tf.contrib.opt.clip_gradients_by_global_norm(
    gradvars, 15.0)
  train_op = multitask_momentum_optimizer.apply_gradients(
    gradvars_clipped, global_step=batch)
  ```
  """
  def __init__(self, opt):
    """
    Args:
    opt: an instance of a class that implements tf.train.Optimizer.
    """
    if not isinstance(opt, optimizer.Optimizer):
      raise TypeError(
          "Supplied optimizer must be an instance of tf.train.Optimizer")
    self._opt = opt
    overriden_methods = ('_apply_dense',
                         '_resource_apply_dense',
                         '_apply_sparse',
                         '_resource_apply_sparse')
    for name in overriden_methods:
      fn = getattr(self._opt, name)
      wrapper = _get_wrapper(fn, self._opt)
      setattr(self._opt, name, wrapper)

  def __getattr__(self, name):
    return getattr(self._opt, name)


def clip_gradients_by_global_norm(gradients_variables, clip_norm=20.):
  """Clips gradients of a multitask loss by their global norm.
  Ignores all-zero tensors when computing the global norm.

  Args:
  gradients_variables: a list of pairs (gradient, variable).
  clip_norm: a float Tensor, the global norm to clip on. Default is 20.0.

  Returns:
  list: A list of pairs of the same type as gradients_variables,.
  fixed_global_norm: A 0-D (scalar) Tensor representing the global norm.
  """
  gradients, variables = six.moves.zip(*gradients_variables)
  def _replace_nonexisting_grad(grad):
    if grad is None:
      return grad
    all_zeros = _is_all_zeros(grad)
    return control_flow_ops.cond(all_zeros,
                                 lambda: array_ops.zeros(
                                     [], dtype=dtypes.as_dtype(grad.dtype)),
                                 lambda: grad)
  nonzero_gradients = [_replace_nonexisting_grad(g) for g in gradients]
  fixed_global_norm = clip_ops.global_norm(nonzero_gradients)
  gradients, _ = clip_ops.clip_by_global_norm(gradients, clip_norm,
                                              use_norm=fixed_global_norm)
  return list(six.moves.zip(gradients, variables)), fixed_global_norm
