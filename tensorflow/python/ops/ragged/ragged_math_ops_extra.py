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
"""Additional math ops for ragged tensors that require extra dependencies.

Most implementations should go in ragged_math_ops.py.  This file is for
implementations that would otherwise introduce circular dependencies -
for example, custom_gradient, which itself depends on some math ops.
"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import math_ops_extra
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch

#===============================================================================
# ragged_reduce_std
#===============================================================================
_RAGGED_REDUCE_STD_EXAMPLE = """
    >>> rt = tf.ragged.constant([[1, 0], [2, 1], [3], [4, 1]],
    ...                         dtype=tf.float64)
    >>> tf.math.reduce_std(rt, axis=0).numpy()
    array([1.11803399, 0.47140452])
    >>> tf.math.reduce_std(rt, axis=1).numpy()
    array([0.5, 0.5, 0., 1.5])
"""


@dispatch.dispatch_for_api(math_ops_extra.reduce_std)
def reduce_std(
    input_tensor: ragged_tensor.Ragged, axis=None, keepdims=False, name=None
):
  """For docs, see: _RAGGED_REDUCE_DOCSTRING."""
  with ops.name_scope(name, 'RaggedReduceStd', [input_tensor, axis]):
    variance = ragged_math_ops.reduce_variance(
        input_tensor, axis=axis, keepdims=keepdims
    )

    # Since the gradient of standard deviation is bounded as variance approaches
    # zero, return zero at the singularity of dsqrt(v)/dv.
    @custom_gradient.custom_gradient
    def safe_sqrt(x):
      y = math_ops.sqrt(x)
      def grad(g):
        return 0.5 * math_ops.div_no_nan(g, y)
      return y, grad

    return safe_sqrt(variance)


ragged_math_ops._set_ragged_reduce_docstring(  # pylint: disable=protected-access
    reduce_std, 'std', 'averaged', 'NaN', _RAGGED_REDUCE_STD_EXAMPLE
)
