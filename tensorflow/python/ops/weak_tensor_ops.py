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
"""Support for WeakTensor in TF ops."""

import inspect

from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import weak_tensor_ops_list
from tensorflow.python.util import dispatch


# This file must depend on math_ops so that e.g. `__add__` is
# added to the Tensor class.
for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
  tensor_oper = getattr(ops.Tensor, operator)
  setattr(WeakTensor, operator, tensor_oper)

# List of unary ops that have support for WeakTensor.
_TF_UNARY_APIS = weak_tensor_ops_list.ALL_UNARY_OPS


def register_unary_weak_tensor_dispatcher(op):
  """Add dispatch for WeakTensor inputs."""
  signature = inspect.signature(op)
  weak_tensor_arg_name = next(iter(signature.parameters.keys()))

  @dispatch.dispatch_for_api(op, {weak_tensor_arg_name: WeakTensor})
  def wrapper(*args, **kwargs):
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    bound_kwargs = bound_arguments.arguments
    bound_kwargs[weak_tensor_arg_name] = bound_kwargs[
        weak_tensor_arg_name
    ].to_tensor()

    # Only return WeakTensor if there is no dtype specified.
    if bound_kwargs.get("dtype", None) is None:
      return WeakTensor.from_tensor((op(**bound_kwargs)))
    else:
      return op(**bound_kwargs)

  return wrapper


for tf_api in _TF_UNARY_APIS:
  register_unary_weak_tensor_dispatcher(tf_api)
