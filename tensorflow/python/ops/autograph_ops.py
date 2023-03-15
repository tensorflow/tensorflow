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
"""Autograph specific overrides for objects covered by tensor_util.is_tf_type."""

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_ops


def wrap_py_func(f, args, kwargs=None):
  """Helper that wraps a callable to py_func.

  The helper passes tensor arguments through the py_func interface. Non-tensor
  arguments are allowed, and will be passed to f directly. Note that non-tensor
  arguments are captured by f will not update every time the wrapper is
  called (this is consistent with its argument list, which only includes
  the tensor arguments). In general, it's safest not to reuse this wrapper.

  Args:
    f: Callable
    args: Positional arguments for f, as list or tuple.
    kwargs: Keyword arguments for f, as dict with string keys. May be None.

  Returns:
    The return values of f converted to tensor.
  Raises:
    ValueError: if any of the arguments are incorrect.
  """
  tensor_args = []
  tensor_args_idx = {}

  # Of the positional arguments, only grab the tensor ones to be passed through
  # the py_func.
  n_args = len(args)
  arg_is_tensor = tuple(map(tensor_util.is_tf_type, args))
  for i in range(n_args):
    if arg_is_tensor[i]:
      tensor_args_idx[i] = len(tensor_args)
      tensor_args.append(args[i])

  # We essentially take the tensor kwargs, if any, and add them to the list of
  # positional arguments. The kwargs are then reconstructed inside the py_func.
  #
  # For example, if
  #
  #     args = [Tensor(1), 'foo']
  #     kwargs = {'a': Tensor(2), 'b': 'bar'}
  #
  # Then
  #
  #     tensor_args = (Tensor(1), Tensor(2))
  #     kwarg_keys = ('a', 'b')
  if kwargs:
    kwarg_keys = tuple(kwargs.keys())
    kwarg_is_tensor = {k: tensor_util.is_tf_type(kwargs[k]) for k in kwarg_keys}
    for k in kwarg_keys:
      if kwarg_is_tensor[k]:
        tensor_args_idx[k] = len(tensor_args)
        tensor_args.append(kwargs[k])
  else:
    kwarg_keys = ()

  def f_wrapper(*tensor_args):
    f_args = tuple(
        tensor_args[tensor_args_idx[i]] if arg_is_tensor[i] else a
        for i, a in enumerate(args)
    )
    f_kwargs = {
        k: tensor_args[tensor_args_idx[k]] if kwarg_is_tensor[k] else kwargs[k]
        for i, k in enumerate(kwarg_keys)
    }
    f(*f_args, **f_kwargs)
    return 1

  return script_ops.eager_py_func(f_wrapper, tensor_args, dtypes.int32)


def _tf_py_func_print(*objects, **kwargs):
  """Overload of print_ as a py_func implementation."""
  override_kwargs = {
      k: v for k, v in kwargs.items() if v is not py_builtins.UNSPECIFIED
  }
  if 'flush' not in override_kwargs:
    # Defaulting to flushing the console in graph mode, which helps reduce
    # garbled output in IPython.
    override_kwargs['flush'] = True

  def print_wrapper(*vals, **kwargs):
    vals = tuple(v.numpy() if tensor_util.is_tf_type(v) else v for v in vals)
    # TensorFlow doesn't seem to generate Unicode when passing strings to
    # py_func. This causes the print to add a "b'" wrapper to the output,
    # which is probably never what you want.
    vals = tuple(v.decode('utf-8') if isinstance(v, bytes) else v for v in vals)
    print(*vals, **kwargs)

  return wrap_py_func(print_wrapper, objects, override_kwargs)


def _tf_sorted(iterable, key, reverse):
  """Overload of sorted_ for Tensor iterable."""
  if reverse is py_builtins.UNSPECIFIED:
    direction = 'ASCENDING'
  else:
    direction = 'DESCENDING'
  if key is not py_builtins.UNSPECIFIED:
    mapped = parallel_ops.vectorized_map(key, iterable)
    if mapped.shape.rank is not None and mapped.shape.rank != 1:
      raise ValueError('sort only supports only 1D tensors')
    with ops.control_dependencies([
        check_ops.assert_rank_v2(mapped, 1,
                                 'sort only supports only 1D tensors')
    ]):
      order = sort_ops.argsort(mapped, direction=direction)
      return array_ops.gather_v2(iterable, order)
  if iterable.shape.rank is not None and iterable.shape.rank != 1:
    raise ValueError('sort only supports only 1D tensors')
  with ops.control_dependencies([
      check_ops.assert_rank_v2(iterable, 1,
                               'sort only supports only 1D tensors')
  ]):
    return sort_ops.sort(iterable, direction=direction)

py_builtins.print_registry.register(
    tensor_util.tf_type_classes, _tf_py_func_print
)
py_builtins.sorted_registry.register(
    tensor_util.tf_type_classes, _tf_sorted
)
