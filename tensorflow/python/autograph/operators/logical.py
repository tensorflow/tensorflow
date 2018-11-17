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
"""Logical operators, including comparison and bool operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops


def not_(a):
  """Functional form of "not"."""
  if tensor_util.is_tensor(a):
    return gen_math_ops.logical_not(a)
  return not a


def and_(a, b):
  """Functional form of "and". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tensor(a_val):
    return _tf_lazy_and(a_val, b)
  return _py_lazy_and(a_val, b)


def _tf_lazy_and(cond, b):
  """Lazy-eval equivalent of "and" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return control_flow_ops.cond(cond, b, lambda: cond)


def _py_lazy_and(cond, b):
  """Lazy-eval equivalent of "and" in Python."""
  return cond and b()


def or_(a, b):
  """Functional form of "or". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tensor(a_val):
    return _tf_lazy_or(a_val, b)
  return _py_lazy_or(a_val, b)


def _tf_lazy_or(cond, b):
  """Lazy-eval equivalent of "or" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return control_flow_ops.cond(cond, lambda: cond, b)


def _py_lazy_or(cond, b):
  """Lazy-eval equivalent of "or" in Python."""
  return cond or b()


def eq(a, b):
  """Functional form of "equal"."""
  if tensor_util.is_tensor(a) or tensor_util.is_tensor(b):
    return _tf_equal(a, b)
  return _py_equal(a, b)


def _tf_equal(a, b):
  """Overload of "equal" for Tensors."""
  return gen_math_ops.equal(a, b)


def _py_equal(a, b):
  """Overload of "equal" that falls back to Python's default implementation."""
  return a == b


def not_eq(a, b):
  """Functional form of "not-equal"."""
  return not_(eq(a, b))


# Default implementation for the remainings.


def gt(a, b):
  """Functional form of "less-than"."""
  return a > b


def gt_e(a, b):
  """Functional form of "less-than"."""
  return a >= b


def is_(a, b):
  """Functional form of "less-than"."""
  return a is b


def is_not(a, b):
  """Functional form of "less-than"."""
  return a is not b


def in_(a, b):
  """Functional form of "less-than"."""
  # TODO(mdan): in and not_in should probably be convertible for some types.
  return a in b


def lt(a, b):
  """Functional form of "less-than"."""
  return a < b


def lt_e(a, b):
  """Functional form of "less-than"."""
  return a <= b


def not_in(a, b):
  """Functional form of "less-than"."""
  return a not in b


def u_sub(a):
  """Functional form of "unary-sub"."""
  return -a
