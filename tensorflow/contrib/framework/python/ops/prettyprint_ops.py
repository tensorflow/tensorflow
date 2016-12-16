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

"""Functions to provide simpler and prettier logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables

__all__ = ["print_op"]


def _get_tensor_repr(t,
                     print_tensor_name=True,
                     print_tensor_type=True,
                     print_shape=True,
                     summarize_indicator_vector=True):
  """Return a list of Tensors that summarize the given tensor t."""
  tensor_list = []
  if print_tensor_name and isinstance(t, ops.Tensor):
    tensor_list.append(constant_op.constant("Name: " + t.name))

  if print_tensor_type:
    if isinstance(t, ops.Tensor):
      t_type_str = "Type: Tensor ({})".format(t.dtype.name)
    elif isinstance(t, sparse_tensor.SparseTensor):
      t_type_str = "Type: SparseTensor ({})".format(t.dtype.name)
    elif isinstance(t, tensor_array_ops.TensorArray):
      t_type_str = "Type: TensorArray ({})".format(t.dtype.name)
    elif isinstance(t, variables.Variable):
      t_type_str = "Type: Variable ({})".format(t.dtype.name)
    else:
      raise ValueError("t must be a Tensor, SparseTensor, TensorArray or "
                       "Variable.")

    tensor_list.append(constant_op.constant(t_type_str))

  if print_shape:
    if isinstance(t, sparse_tensor.SparseTensor):
      tensor_list.append(constant_op.constant("Shape:"))
      tensor_list.append(t.dense_shape)
    elif isinstance(t, ops.Tensor):
      tensor_list.append(constant_op.constant("Shape: " + str(t.get_shape(
      ).dims)))
    elif isinstance(t, tensor_array_ops.TensorArray):
      tensor_list.append(constant_op.constant("Size:"))
      tensor_list.append(t.size())

  if summarize_indicator_vector and t.dtype == dtypes.bool:
    int_tensor = math_ops.cast(t, dtypes.uint8)
    tensor_list.append(constant_op.constant("First True in Boolean tensor at:"))
    tensor_list.append(math_ops.argmax(int_tensor, 0))

  if isinstance(t, sparse_tensor.SparseTensor):
    tensor_list.append(constant_op.constant("Sparse indices:"))
    tensor_list.append(t.indices)
    tensor_list.append(constant_op.constant("Sparse values:"))
    tensor_list.append(t.values)
  elif isinstance(t, ops.Tensor):
    tensor_list.append(constant_op.constant("Value:"))
    tensor_list.append(t)
  elif isinstance(t, tensor_array_ops.TensorArray):
    tensor_list.append(constant_op.constant("Value:"))
    tensor_list.append(t.stack())

  return tensor_list


def print_op(input_,
             data=None,
             message=None,
             first_n=None,
             summarize=20,
             print_tensor_name=True,
             print_tensor_type=True,
             print_shape=True,
             summarize_indicator_vector=True,
             name=None):
  """Creates a print op that will print when a tensor is accessed.

  Wraps the tensor passed in so that whenever that tensor is accessed,
  the message `message` is printed, along with the current value of the
  tensor `t` and an optional list of other tensors.

  Args:
    input_: A Tensor/SparseTensor/TensorArray to print when it is evaluated.
    data: A list of other tensors to print.
    message: A string message to print as a prefix.
    first_n: Only log `first_n` number of times. Negative numbers log always;
             this is the default.
    summarize: Print this number of elements in the tensor.
    print_tensor_name: Print the tensor name.
    print_tensor_type: Print the tensor type.
    print_shape: Print the tensor's shape.
    summarize_indicator_vector: Whether to print the index of the first true
      value in an indicator vector (a Boolean tensor).
    name: The name to give this op.

  Returns:
    A Print op. The Print op returns `input_`.

  Raises:
    ValueError: If the tensor `input_` is not a Tensor, SparseTensor or
      TensorArray.

  """

  message = message or ""
  if input_ is None:
    raise ValueError("input_ must be of type "
                     "Tensor, SparseTensor or TensorArray")

  tensor_list = _get_tensor_repr(input_, print_tensor_name, print_tensor_type,
                                 print_shape, summarize_indicator_vector)

  if data is not None:
    for t in data:
      tensor_list.extend(_get_tensor_repr(t, print_tensor_name,
                                          print_tensor_type, print_shape,
                                          summarize_indicator_vector))

  if isinstance(input_, ops.Tensor) or isinstance(input_, variables.Variable):
    input_ = logging_ops.Print(input_, tensor_list, message, first_n, summarize,
                               name)
  elif isinstance(input_, sparse_tensor.SparseTensor):
    p = logging_ops.Print(
        constant_op.constant([]), tensor_list, message, first_n, summarize,
        name)

    with ops.control_dependencies([p]):
      input_ = sparse_tensor.SparseTensor(
          array_ops.identity(input_.indices),
          array_ops.identity(input_.values),
          array_ops.identity(input_.dense_shape))
  elif isinstance(input_, tensor_array_ops.TensorArray):
    p = logging_ops.Print(
        constant_op.constant([]), tensor_list, message, first_n, summarize,
        name)

    with ops.control_dependencies([p]):
      input_ = tensor_array_ops.TensorArray(dtype=input_.dtype,
                                            handle=input_.handle)
  else:
    raise ValueError("input_ must be of type "
                     "Tensor, SparseTensor or TensorArray")

  return input_
