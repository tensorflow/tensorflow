# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Registration and usage mechanisms for KL-divergences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops


_DIVERGENCES = {}


def kl(dist_a, dist_b, allow_nan=False, name=None):
  """Get the KL-divergence KL(dist_a || dist_b).

  Args:
    dist_a: instance of distributions.Distribution.
    dist_b: instance of distributions.Distribution.
    allow_nan: If `False` (default), a runtime error is raised
      if the KL returns NaN values for any batch entry of the given
      distributions.  If `True`, the KL may return a NaN for the given entry.
    name: (optional) Name scope to use for created operations.

  Returns:
    A Tensor with the batchwise KL-divergence between dist_a and dist_b.

  Raises:
    TypeError: If dist_a or dist_b is not an instance of Distribution.
    NotImplementedError: If no KL method is defined for distribution types
      of dist_a and dist_b.
  """
  if not isinstance(dist_a, distribution.Distribution):
    raise TypeError(
        "dist_a is not an instance of Distribution, received type: %s"
        % type(dist_a))
  if not isinstance(dist_b, distribution.Distribution):
    raise TypeError(
        "dist_b is not an instance of Distribution, received type: %s"
        % type(dist_b))
  kl_fn = _DIVERGENCES.get((type(dist_a), type(dist_b)), None)
  if kl_fn is None:
    raise NotImplementedError(
        "No KL(dist_a || dist_b) registered for dist_a type %s and dist_b "
        "type %s" % ((type(dist_a).__name__, type(dist_b).__name__)))
  with ops.name_scope("KullbackLeibler"):
    kl_t = kl_fn(dist_a, dist_b, name=name)
    if allow_nan:
      return kl_t

    # Check KL for NaNs
    kl_t = array_ops.identity(kl_t, name="kl")

    with ops.control_dependencies([
        logging_ops.Assert(
            math_ops.logical_not(
                math_ops.reduce_any(math_ops.is_nan(kl_t))),
            ["KL calculation between %s and %s returned NaN values "
             "(and was called with allow_nan=False).  Values:"
             % (dist_a.name, dist_b.name), kl_t])]):
      return array_ops.identity(kl_t, name="checked_kl")


class RegisterKL(object):
  """Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  """

  def __init__(self, dist_cls_a, dist_cls_b):
    """Initialize the KL registrar.

    Args:
      dist_cls_a: the class of the first argument of the KL divergence.
      dist_cls_b: the class of the second argument of the KL divergence.

    Raises:
      TypeError: if dist_cls_a or dist_cls_b are not subclasses of
        Distribution.
    """

    if not issubclass(dist_cls_a, distribution.Distribution):
      raise TypeError("%s is not a subclass of Distribution" % dist_cls_a)
    if not issubclass(dist_cls_b, distribution.Distribution):
      raise TypeError("%s is not a subclass of Distribution" % dist_cls_b)
    self._key = (dist_cls_a, dist_cls_b)

  def __call__(self, kl_fn):
    """Perform the KL registration.

    Args:
      kl_fn: The function to use for the KL divergence.

    Returns:
      kl_fn

    Raises:
      TypeError: if kl_fn is not a callable.
      ValueError: if a KL divergence function has already been registered for
        the given argument classes.
    """
    if not callable(kl_fn):
      raise TypeError("kl_fn must be callable, received: %s" % kl_fn)
    if self._key in _DIVERGENCES:
      raise ValueError("KL(%s || %s) has already been registered to: %s"
                       % (self._key[0].__name__, self._key[1].__name__,
                          _DIVERGENCES[self._key]))
    _DIVERGENCES[self._key] = kl_fn
    return kl_fn
