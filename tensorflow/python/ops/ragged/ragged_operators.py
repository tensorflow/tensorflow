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
"""Operator overloads for `RaggedTensor`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.ragged import ragged_elementwise_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator


def _right(operator):
  """Right-handed version of an operator: swap args x and y."""
  return tf_decorator.make_decorator(operator, lambda y, x: operator(x, y))


# Indexing
ragged_tensor.RaggedTensor.__getitem__ = ragged_getitem.ragged_tensor_getitem

# Ordering operators
ragged_tensor.RaggedTensor.__ge__ = ragged_elementwise_ops.greater_equal
ragged_tensor.RaggedTensor.__gt__ = ragged_elementwise_ops.greater
ragged_tensor.RaggedTensor.__le__ = ragged_elementwise_ops.less_equal
ragged_tensor.RaggedTensor.__lt__ = ragged_elementwise_ops.less

# Logical operators
ragged_tensor.RaggedTensor.__and__ = ragged_elementwise_ops.logical_and
ragged_tensor.RaggedTensor.__rand__ = _right(ragged_elementwise_ops.logical_and)
ragged_tensor.RaggedTensor.__invert__ = ragged_elementwise_ops.logical_not
ragged_tensor.RaggedTensor.__ror__ = _right(ragged_elementwise_ops.logical_or)
ragged_tensor.RaggedTensor.__or__ = ragged_elementwise_ops.logical_or
ragged_tensor.RaggedTensor.__xor__ = ragged_elementwise_ops.logical_xor
ragged_tensor.RaggedTensor.__rxor__ = _right(ragged_elementwise_ops.logical_xor)

# Arithmetic operators
ragged_tensor.RaggedTensor.__abs__ = ragged_elementwise_ops.abs
ragged_tensor.RaggedTensor.__add__ = ragged_elementwise_ops.add
ragged_tensor.RaggedTensor.__radd__ = _right(ragged_elementwise_ops.add)
ragged_tensor.RaggedTensor.__div__ = ragged_elementwise_ops.div
ragged_tensor.RaggedTensor.__rdiv__ = _right(ragged_elementwise_ops.div)
ragged_tensor.RaggedTensor.__floordiv__ = ragged_elementwise_ops.floordiv
ragged_tensor.RaggedTensor.__rfloordiv__ = _right(
    ragged_elementwise_ops.floordiv)
ragged_tensor.RaggedTensor.__mod__ = ragged_elementwise_ops.floormod
ragged_tensor.RaggedTensor.__rmod__ = _right(ragged_elementwise_ops.floormod)
ragged_tensor.RaggedTensor.__mul__ = ragged_elementwise_ops.multiply
ragged_tensor.RaggedTensor.__rmul__ = _right(ragged_elementwise_ops.multiply)
ragged_tensor.RaggedTensor.__neg__ = ragged_elementwise_ops.negative
ragged_tensor.RaggedTensor.__pow__ = ragged_elementwise_ops.pow
ragged_tensor.RaggedTensor.__rpow__ = _right(ragged_elementwise_ops.pow)
ragged_tensor.RaggedTensor.__sub__ = ragged_elementwise_ops.subtract
ragged_tensor.RaggedTensor.__rsub__ = _right(ragged_elementwise_ops.subtract)
ragged_tensor.RaggedTensor.__truediv__ = ragged_elementwise_ops.truediv
ragged_tensor.RaggedTensor.__rtruediv__ = _right(ragged_elementwise_ops.truediv)


# Dummy methods
def _dummy_bool(_):
  """Dummy method to prevent a RaggedTensor from being used as a Python bool."""
  raise TypeError("RaggedTensor may not be used as a boolean.")


ragged_tensor.RaggedTensor.__bool__ = _dummy_bool
ragged_tensor.RaggedTensor.__nonzero__ = _dummy_bool
