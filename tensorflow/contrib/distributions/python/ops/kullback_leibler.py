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

import inspect

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


_DIVERGENCES = {}


def _registered_kl(type_a, type_b):
  """Get the KL function registered for classes a and b."""
  hierarchy_a = inspect.getmro(type_a)
  hierarchy_b = inspect.getmro(type_b)
  dist_to_children = None
  kl_fn = None
  for mro_to_a, parent_a in enumerate(hierarchy_a):
    for mro_to_b, parent_b in enumerate(hierarchy_b):
      candidate_dist = mro_to_a + mro_to_b
      candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
      if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
        dist_to_children = candidate_dist
        kl_fn = candidate_kl_fn
  return kl_fn


def kl(dist_a, dist_b, allow_nan_stats=True, name=None):
  """Get the KL-divergence KL(dist_a || dist_b).

  If there is no KL method registered specifically for `type(dist_a)` and
  `type(dist_b)`, then the class hierarchies of these types are searched.

  If one KL method is registered between any pairs of classes in these two
  parent hierarchies, it is used.

  If more than one such registered method exists, the method whose registered
  classes have the shortest sum MRO paths to the input types is used.

  If more than one such shortest path exists, the first method
  identified in the search is used (favoring a shorter MRO distance to
  `type(dist_a)`).

  Args:
    dist_a: The first distribution.
    dist_b: The second distribution.
    allow_nan_stats: Python `bool`, default `True`. When `True`,
      statistics (e.g., mean, mode, variance) use the value "`NaN`" to
      indicate the result is undefined. When `False`, an exception is raised
      if one or more of the statistic's batch members are undefined.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    A Tensor with the batchwise KL-divergence between dist_a and dist_b.

  Raises:
    NotImplementedError: If no KL method is defined for distribution types
      of dist_a and dist_b.
  """
  kl_fn = _registered_kl(type(dist_a), type(dist_b))
  if kl_fn is None:
    raise NotImplementedError(
        "No KL(dist_a || dist_b) registered for dist_a type %s and dist_b "
        "type %s" % (type(dist_a).__name__, type(dist_b).__name__))

  with ops.name_scope("KullbackLeibler"):
    kl_t = kl_fn(dist_a, dist_b, name=name)
    if allow_nan_stats:
      return kl_t

    # Check KL for NaNs
    kl_t = array_ops.identity(kl_t, name="kl")

    with ops.control_dependencies([
        control_flow_ops.Assert(
            math_ops.logical_not(
                math_ops.reduce_any(math_ops.is_nan(kl_t))),
            ["KL calculation between %s and %s returned NaN values "
             "(and was called with allow_nan_stats=False). Values:"
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
    """
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
