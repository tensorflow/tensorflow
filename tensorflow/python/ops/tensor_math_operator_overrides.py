# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Overrides for Tensor operators."""


from tensorflow.python.framework import override_binary_operator
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import tf_decorator


# pylint: disable=g-import-not-at-top
def _add_dispatch_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops._add_dispatch(x, y, name=name)  # pylint: disable=protected-access


def _and_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.and_(x, y, name=name)


def _div_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.div(x, y, name=name)


def _floordiv_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.floordiv(x, y, name=name)


def _matmul_factory(a, b, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.matmul_wrapper(a, b, name=name)


def _mod_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.mod(x, y, name=name)


def _mul_dispatch_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops._mul_dispatch(x, y, name=name)  # pylint: disable=protected-access


def _or_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.or_(x, y, name=name)


def _pow_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.pow(x, y, name=name)


def _subtract_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.subtract(x, y, name=name)


def _truediv_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.truediv(x, y, name=name)


def _xor_factory(x, y, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.xor_(x, y, name=name)


override_binary_operator.override_binary_operator_helper(
    _add_dispatch_factory, "add"
)
override_binary_operator.override_binary_operator_helper(_and_factory, "and")
override_binary_operator.override_binary_operator_helper(_div_factory, "div")
override_binary_operator.override_binary_operator_helper(
    _floordiv_factory, "floordiv"
)
override_binary_operator.override_binary_operator_helper(
    _matmul_factory, "matmul"
)
override_binary_operator.override_binary_operator_helper(_mod_factory, "mod")
override_binary_operator.override_binary_operator_helper(
    _mul_dispatch_factory, "mul"
)
override_binary_operator.override_binary_operator_helper(_or_factory, "or")
override_binary_operator.override_binary_operator_helper(_pow_factory, "pow")
override_binary_operator.override_binary_operator_helper(
    _subtract_factory, "sub"
)
override_binary_operator.override_binary_operator_helper(
    _truediv_factory, "truediv"
)
override_binary_operator.override_binary_operator_helper(_xor_factory, "xor")


def _invert_factory(x, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.invert_(x, name=name)


def _abs_factory(x, name=None):
  from tensorflow.python.ops import math_ops

  return math_ops.abs(x, name=name)


def _tensor_equals_factory(self, other):
  from tensorflow.python.ops import math_ops

  return math_ops.tensor_equals(self, other)


def _tensor_not_equals_factory(self, other):
  from tensorflow.python.ops import math_ops

  return math_ops.tensor_not_equals(self, other)


def _promote_dtypes_decorator(fn):
  def wrapper(x, y, *args, **kwargs):
    x, y = override_binary_operator.maybe_promote_tensors(x, y)
    return fn(x, y, *args, **kwargs)

  return tf_decorator.make_decorator(fn, wrapper)


# pylint: disable=protected-access
tensor_lib.Tensor._override_operator("__invert__", _invert_factory)
tensor_lib.Tensor._override_operator("__neg__", gen_math_ops.neg)
tensor_lib.Tensor._override_operator("__abs__", _abs_factory)
tensor_lib.Tensor._override_operator("__lt__", _promote_dtypes_decorator(
    gen_math_ops.less))
tensor_lib.Tensor._override_operator("__le__", _promote_dtypes_decorator(
    gen_math_ops.less_equal))
tensor_lib.Tensor._override_operator("__gt__", _promote_dtypes_decorator(
    gen_math_ops.greater))
tensor_lib.Tensor._override_operator("__ge__", _promote_dtypes_decorator(
    gen_math_ops.greater_equal))
tensor_lib.Tensor._override_operator("__eq__", _tensor_equals_factory)
tensor_lib.Tensor._override_operator("__ne__", _tensor_not_equals_factory)
