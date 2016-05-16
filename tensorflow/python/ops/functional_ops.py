# Copyright 2015 Google Inc. All Rights Reserved.
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
# =============================================================================

"""Functional operations.

## Higher Order Operators

TensorFlow provides several higher order operators to simplify the common
map-reduce programming patterns.

@@map_fn
@@foldl
@@foldr
@@scan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_functional_ops import *
# pylint: enable=wildcard-import
# pylint: disable=unused-import
from tensorflow.python.ops.gen_functional_ops import _symbolic_gradient
# pylint: enable=unused-import


# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
  """foldl on the list of tensors unpacked from `elems` on dimension 0.

  This foldl operator repeatedly applies the callable `fn` to a sequence
  of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn. If `initializer` is None, `elems` must contain
  at least one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is fn(initializer, values[0]).shape`.

  Args:
    fn: The callable to be performed.
    elems: A tensor to be unpacked on dimension 0.
    initializer: (optional) The initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run
                         in parallel.
    back_prop: (optional) True enables back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor resulting from applying `fn` consecutively to the list of tensors
    unpacked from `elems`, from first to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldl(lambda a, x: a + x, elems)
    # sum == 21
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  # TODO(ebrevdo): Change to using colocate_with here and in other methods.
  with vs.variable_op_scope([elems], name, "foldl") as varscope:
    # Any get_variable calls fn will cache the first call locally
    # and not issue repeated network I/O requests for each iteration.
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Convert elems to tensor array.
    elems = ops.convert_to_tensor(elems, name="elems")
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False,
                                            infer_shape=True)
    elems_ta = elems_ta.unpack(elems)

    if initializer is None:
      a = elems_ta.read(0)
      i = constant_op.constant(1)
    else:
      a = ops.convert_to_tensor(initializer)
      i = constant_op.constant(0)

    def compute(i, a):
      a = fn(a, elems_ta.read(i))
      return [i + 1, a]
    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i < n, compute, [i, a],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    return r_a


def foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
  """foldr on the list of tensors unpacked from `elems` on dimension 0.

  This foldr operator repeatedly applies the callable `fn` to a sequence
  of elements from last to first. The elements are made of the tensors
  unpacked from `elems`. The callable fn takes two tensors as arguments.
  The first argument is the accumulated value computed from the preceding
  invocation of fn. If `initializer` is None, `elems` must contain at least
  one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `fn(initializer, values[0]).shape`.

  Args:
    fn: The callable to be performed.
    elems: A tensor that is unpacked into a sequence of tensors to apply `fn`.
    initializer: (optional) The initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run
                         in parallel.
    back_prop: (optional) True enables back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor resulting from applying `fn` consecutively to the list of tensors
    unpacked from `elems`, from last to first.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldr(lambda a, x: a + x, elems)
    # sum == 21
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  with vs.variable_op_scope([elems], name, "foldr") as varscope:
    # Any get_variable calls fn will cache the first call locally
    # and not issue repeated network I/O requests for each iteration.
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Convert elems to tensor array.
    elems = ops.convert_to_tensor(elems, name="elems")
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False,
                                            infer_shape=True)
    elems_ta = elems_ta.unpack(elems)

    if initializer is None:
      i = n - 1
      a = elems_ta.read(i)
    else:
      i = n
      a = ops.convert_to_tensor(initializer)
    def compute(i, a):
      i -= 1
      a = fn(a, elems_ta.read(i))
      return [i, a]
    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i > 0, compute, [i, a],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    return r_a


def map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True,
           swap_memory=False, name=None):
  """map on the list of tensors unpacked from `elems` on dimension 0.

  This map operator repeatedly applies the callable `fn` to a sequence of
  elements from first to last. The elements are made of the tensors unpacked
  from `elems`. `dtype` is the data type of the return value of `fn`. Users
  must provide `dtype` if it is different from the data type of `elems`.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[len(values)] + fn(values[0]).shape`.

  Args:
    fn: The callable to be performed.
    elems: A tensor to be unpacked to apply `fn`.
    dtype: (optional) The output type of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run
                         in parallel.
    back_prop: (optional) True enables back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor that packs the results of applying `fn` to the list of tensors
    unpacked from `elems`, from first to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    squares = map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  with vs.variable_op_scope([elems], name, "map") as varscope:
    # Any get_variable calls fn will cache the first call locally
    # and not issue repeated network I/O requests for each iteration.
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    elems = ops.convert_to_tensor(elems, name="elems")
    dtype = dtype if dtype else elems.dtype

    # Convert elems to tensor array.
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False,
                                            infer_shape=True)
    elems_ta = elems_ta.unpack(elems)

    i = constant_op.constant(0)
    acc_ta = tensor_array_ops.TensorArray(dtype=dtype, size=n,
                                          dynamic_size=False,
                                          infer_shape=True)
    def compute(i, ta):
      ta = ta.write(i, fn(elems_ta.read(i)))
      return [i + 1, ta]
    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i < n, compute, [i, acc_ta],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    result = r_a.pack()
    result.set_shape(elems.get_shape().with_rank_at_least(1)[0:1].concatenate(
        result.get_shape()[1:]))
    return result


def scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
         swap_memory=False, name=None):
  """scan on the list of tensors unpacked from `elems` on dimension 0.

  This scan operator repeatedly applies the callable `fn` to a sequence
  of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn. If `initializer` is None, `elems` must contain
  at least one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`.

  Args:
    fn: The callable to be performed.
    elems: A tensor to be unpacked on dimension 0.
    initializer: (optional) The initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run
                         in parallel.
    back_prop: (optional) True enables back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor that packs the results of applying `fn` to the list of tensors
    unpacked from `elems`, from first to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = scan(lambda a, x: a + x, elems)
    # sum == [1, 3, 6, 10, 15, 21]
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  with vs.variable_op_scope([elems], name, "scan") as varscope:
    # Any get_variable calls fn will cache the first call locally
    # and not issue repeated network I/O requests for each iteration.
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Convert elems to tensor array.
    elems = ops.convert_to_tensor(elems, name="elems")
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False,
                                            infer_shape=True)
    elems_ta = elems_ta.unpack(elems)

    if initializer is None:
      a = elems_ta.read(0)
      i = constant_op.constant(1)
    else:
      a = ops.convert_to_tensor(initializer)
      i = constant_op.constant(0)

    # Create a tensor array to store the intermediate values.
    acc_ta = tensor_array_ops.TensorArray(dtype=a.dtype, size=n,
                                          dynamic_size=False,
                                          infer_shape=True)
    if initializer is None:
      acc_ta = acc_ta.write(0, a)

    def compute(i, a, ta):
      a = fn(a, elems_ta.read(i))
      ta = ta.write(i, a)
      return [i + 1, a, ta]
    _, _, r_a = control_flow_ops.while_loop(
        lambda i, a, ta: i < n, compute, [i, a, acc_ta],
        parallel_iterations=parallel_iterations,
        back_prop=back_prop, swap_memory=swap_memory)
    result = r_a.pack()
    result.set_shape(elems.get_shape().with_rank_at_least(1)[0:1].concatenate(
        result.get_shape()[1:]))
    return result


@ops.RegisterShape("SymbolicGradient")
def _symbolic_gradient_shape(op):
  # Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
  # (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
  # outputs (dx, dy, dz) are the same as (x, y, z).
  return [op.inputs[i].get_shape() for i in range(len(op.outputs))]
