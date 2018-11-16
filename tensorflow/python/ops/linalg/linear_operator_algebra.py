# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Registration mechanisms for various n-ary operations on LinearOperators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect


_CHOLESKY_DECOMPS = {}


def _registered_cholesky(type_a):
  """Get the Cholesky function registered for class a."""
  hierarchy_a = tf_inspect.getmro(type_a)
  distance_to_children = None
  cholesky_fn = None
  for mro_to_a, parent_a in enumerate(hierarchy_a):
    candidate_dist = mro_to_a
    candidate_cholesky_fn = _CHOLESKY_DECOMPS.get(parent_a, None)
    if not cholesky_fn or (
        candidate_cholesky_fn and candidate_dist < distance_to_children):
      distance_to_children = candidate_dist
      cholesky_fn = candidate_cholesky_fn
  return cholesky_fn


def cholesky(lin_op_a, name=None):
  """Get the Cholesky factor associated to lin_op_a.

  Args:
    lin_op_a: The LinearOperator to decompose.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the lower Cholesky factor of `lin_op_a`.

  Raises:
    NotImplementedError: If no Cholesky method is defined for the LinearOperator
      type of `lin_op_a`.
  """
  cholesky_fn = _registered_cholesky(type(lin_op_a))
  if cholesky_fn is None:
    raise ValueError("No cholesky decomposition registered for {}".format(
        type(lin_op_a)))

  with ops.name_scope(name, "Cholesky"):
    return cholesky_fn(lin_op_a)


class RegisterCholesky(object):
  """Decorator to register a Cholesky implementation function.

  Usage:

  @linear_operator_algebra.RegisterCholesky(lin_op.LinearOperatorIdentity)
  def _cholesky_identity(lin_op_a):
    # Return the identity matrix.
  """

  def __init__(self, lin_op_cls_a):
    """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to decompose.
    """
    self._key = lin_op_cls_a

  def __call__(self, cholesky_fn):
    """Perform the Cholesky registration.

    Args:
      cholesky_fn: The function to use for the Cholesky.

    Returns:
      cholesky_fn

    Raises:
      TypeError: if cholesky_fn is not a callable.
      ValueError: if a Cholesky function has already been registered for
        the given argument classes.
    """
    if not callable(cholesky_fn):
      raise TypeError(
          "cholesky_fn must be callable, received: {}".format(cholesky_fn))
    if self._key in _CHOLESKY_DECOMPS:
      raise ValueError("Cholesky({}) has already been registered to: {}".format(
          self._key.__name__, _CHOLESKY_DECOMPS[self._key]))
    _CHOLESKY_DECOMPS[self._key] = cholesky_fn
    return cholesky_fn
