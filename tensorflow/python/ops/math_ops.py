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
"""Math Operations."""
import builtins
import numpy as np

from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import override_binary_operator
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import tensor_math_operator_overrides  # pylint: disable=unused-import
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

nextafter = gen_math_ops.next_after


@tf_export("math.multiply", "multiply")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply(x, y, name=None):
  """Returns an element-wise x * y."""
  if not tensor_util.is_tf_type(x) and tensor_util.is_tf_type(y):
    x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype)
  elif tensor_util.is_tf_type(x) and not tensor_util.is_tf_type(y):
    y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype)
  return gen_math_ops.mul(x, y, name)


@tf_export("math.pow", "pow")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def pow(x, y, name=None):
  """Computes the power of one value to another."""
  if not tensor_util.is_tf_type(x) and tensor_util.is_tf_type(y):
    x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype)
  elif tensor_util.is_tf_type(x) and not tensor_util.is_tf_type(y):
    y = ops.convert_to_tensor(y, dtype=x.dtype.base_dtype)
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)
