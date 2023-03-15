# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

import inspect

from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops


UNSPECIFIED = object()

abs_registry = type_registry.TypeRegistry()
len_registry = type_registry.TypeRegistry()
print_registry = type_registry.TypeRegistry()
enumerate_registry = type_registry.TypeRegistry()
zip_registry = type_registry.TypeRegistry()
map_registry = type_registry.TypeRegistry()
filter_registry = type_registry.TypeRegistry()
any_registry = type_registry.TypeRegistry()
all_registry = type_registry.TypeRegistry()
sorted_registry = type_registry.TypeRegistry()
next_registry = type_registry.TypeRegistry()


def registry_lookup(reg, obj):
  try:
    return reg.lookup(obj)
  except LookupError:
    pass
  return None


def overload_of(f):
  if f in SUPPORTED_BUILTINS:
    return BUILTIN_FUNCTIONS_MAP[f.__name__]
  return f


def _find_originating_frame(caller_fn_scope, innermost=True):
  """Locates the frame in which `caller_fn_scope` was defined."""
  ctx_frame = inspect.currentframe()
  result = None
  while ctx_frame is not None:
    # Note it should not be normally possible to get false positives this way
    # because the function scope object is not accessible to user code (barring
    # call stack introspection).
    if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
      result = ctx_frame
      if innermost:
        break
    ctx_frame = ctx_frame.f_back

  assert result is not None, (
      'the conversion process should ensure the caller_fn_scope is always'
      ' found somewhere on the call stack')

  return result


def locals_in_original_context(caller_fn_scope):
  """Executes the locals function in the context of a specified function."""
  return _find_originating_frame(caller_fn_scope, innermost=True).f_locals


def globals_in_original_context(caller_fn_scope):
  """Executes the locals function in the context of a specified function."""
  return _find_originating_frame(caller_fn_scope, innermost=True).f_globals


def eval_in_original_context(f, args, caller_fn_scope):
  """Executes the eval function in the context of a specified function."""
  # When control flow is rewritten using functions, eval should use the
  # variables found in the same block where it was called. That is equivalent
  # to the innermost function call.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)

  args = (
      args[0],
      ctx_frame.f_globals if len(args) < 2 else args[1],
      ctx_frame.f_locals if len(args) < 3 else args[2],
  )
  return f(*args)


def super_in_original_context(f, args, caller_fn_scope):
  """Executes the super function in the context of a specified function.

  See https://docs.python.org/3/library/functions.html#super for the exact
  details

  Args:
    f: Callable, typically the super builtin
    args: List[Any], the original call arguments
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made

  Returns:
    The result of calling `f` as if it was called in the frame indicated by
      `caller_fn_scope`.
  """

  # Only the no-arg call is desugared.
  if args:
    return f(*args)

  # Inner functions seem to include their closure in f_locals, so we need
  # to find the outermost frame.
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)

  # When super(..) is called without arguments, it looks for __class__ cell
  # variable and the first argument passed in the enclosing function according
  # to the spec https://www.python.org/dev/peps/pep-3135/ .
  #
  # We couldn't verify if `inspect.currentframe().f_code.co_varnames[0]` is
  # guaranteed to be the first argument from an official doc or PEP, however,
  # it's fairly stable and well established:
  # - An unofficial community doc mentions it.
  #   https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
  # - CPython has tests checking that order, which was merged in 2008, and
  #   unchanged since then.
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py2_test_grammar.py#L157
  #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py3_test_grammar.py#L192
  #
  # Note: the name can be more reliably obtained by inspecting the calling
  # function's argspec.
  #
  # Even though methods can be declared using *args (def method(*args)),
  # that pattern is disallowed by super() -- it raises super() no arguments.
  # Method definitions using **kwargs are not allowed at all.
  # In other words, we can always assume that self is on the first positional
  # argument (for correct code).
  #
  # TODO(mdan): Consider additional checks in case the input code is incorrect.
  # For example, the error might be cryptic compared to what super() regularly
  # raises.

  type_arg = ctx_frame.f_locals['__class__']
  self_arg_name = ctx_frame.f_code.co_varnames[0]
  self_arg = ctx_frame.f_locals[self_arg_name]
  return f(type_arg, self_arg)


def abs_(x):
  abs_override = registry_lookup(abs_registry, x)
  if abs_override is not None:
    return abs_override(x)
  if tensor_util.is_tf_type(x):
    return _tf_abs(x)
  return _py_abs(x)


def _tf_abs(x):
  return math_ops.abs(x)


def _py_abs(x):
  return abs(x)


def float_(x=0):
  if tensor_util.is_tf_type(x):
    return _tf_float(x)
  return _py_float(x)


def _tf_float(x):
  # TODO(mdan): We shouldn't assume float32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.float32)
  return math_ops.cast(x, dtype=dtypes.float32)


def _py_float(x):
  return float(x)


def int_(x=0, base=UNSPECIFIED):
  if tensor_util.is_tf_type(x):
    return _tf_int(x, base)
  return _py_int(x, base)


def _tf_int(x, base):
  if base not in (10, UNSPECIFIED):
    raise NotImplementedError('base {} not supported for int'.format(base))

  # TODO(mdan): We shouldn't assume int32.
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
  return math_ops.cast(x, dtype=dtypes.int32)


def _py_int(x, base):
  if base is UNSPECIFIED:
    return int(x)
  return int(x, base)


def len_(s):
  len_override = registry_lookup(len_registry, s)
  if len_override is not None:
    return len_override(s)
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tf_type(s):
    return _tf_tensor_len(s)
  return _py_len(s)


def _tf_tensor_array_len(s):
  return s.size()


def _tf_tensor_list_len(s):
  return list_ops.tensor_list_length(s)


def _tf_tensor_len(s):
  """Overload of len_ for Tensor arguments."""
  # Statically shaped tensors: length is known ahead of time.
  if s.shape.ndims and s.shape.dims[0].value is not None:
    return s.shape.dims[0].value

  # Static shape of unknown dimensions: use dynamic shape but statically
  # check that it's a scalar.
  shape = array_ops.shape(s)

  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)

  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))

  if shape.shape.dims[0].value is not None:
    return array_ops.shape(s)[0]

  # Fully dynamic shape: use ops.
  rank = array_ops.rank(s)

  def raise_zero_rank_error():
    msg = gen_string_ops.string_join(
        ['len requires non-zero rank, got ',
         gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_assert.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)

  return control_flow_ops.cond(rank > 0, lambda: array_ops.shape(s)[0],
                               raise_zero_rank_error)


def _py_len(s):
  return len(s)


def print_(*objects, **kwargs):
  """Overload of the print builtin."""
  # Note: Python 2.6 doesn't support explicit keywords after starargs.
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))

  print_fn = _py_print
  for x in objects:
    print_override = registry_lookup(print_registry, x)
    if print_override is not None:  # pylint: disable=comparison-with-callable
      print_fn = print_override
      break

  if print_fn is _py_print:
    # If this fails, ops/autograph_ops.py hasn't been imported.
    assert not any(tensor_util.is_tf_type(s) for s in objects)

  return print_fn(*objects, **kwargs)


def _py_print(*objects, **kwargs):
  print(*objects, **kwargs)


def min_(*args, **kwargs):
  if any(tensor_util.is_tf_type(s) for s in args):
    return _tf_min(*args, **kwargs)
  return _py_min(*args, **kwargs)


def _tf_min(*args, **kwargs):
  if len(kwargs):
    kwargs_tuple = tuple(set(kwargs.keys()))
    raise ValueError('These keyword arguments are '
                     'currently not supported: {}'.format(kwargs_tuple))
  if len(args) == 1:
    rank = args[0].shape.rank
    if rank == 0:
      return args[0]
    if rank == 1:
      return math_ops.reduce_min(*args, axis=0)
    raise ValueError('min(arg) currently support only tensor with rank 1, '
                     'but got a tensor with rank {}'.format(rank))
  for arg in args:
    rank = arg.shape.rank
    if rank != 0:
      raise ValueError('min(arg1, arg2, *args) currently support '
                       'only scalar tensor, but got a tensor '
                       'with shape {}'.format(rank))
  return math_ops.reduce_min(args, axis=0)


def _py_min(*args, **kwargs):
  return min(*args, **kwargs)


def max_(*args, **kwargs):
  if any(tensor_util.is_tf_type(s) for s in args):
    return _tf_max(*args, **kwargs)
  return _py_max(*args, **kwargs)


def _tf_max(*args, **kwargs):
  if len(kwargs):
    kwargs_tuple = tuple(set(kwargs.keys()))
    raise ValueError('These keyword arguments are '
                     'currently not supported: {}'.format(kwargs_tuple))
  if len(args) == 1:
    rank = args[0].shape.rank
    if rank == 0:
      return args[0]
    if rank == 1:
      return math_ops.reduce_max(*args, axis=0)
    raise ValueError('max(arg) currently support only tensor with rank 1, '
                     'but got a tensor with rank {}'.format(rank))
  for arg in args:
    rank = arg.shape.rank
    if rank != 0:
      raise ValueError('max(arg1, arg2, *args) currently support '
                       'only scalar tensor, but got a tensor '
                       'with shape {}'.format(rank))
  return math_ops.reduce_max(args, axis=0)


def _py_max(*args, **kwargs):
  return max(*args, **kwargs)


def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
  if any(tensor_util.is_tf_type(s) for s in (start_or_stop, stop, step)):
    return _tf_range(start_or_stop, stop, step)
  return _py_range(start_or_stop, stop, step)


def _tf_range(start_or_stop, stop, step):
  """Overload of range_ that generates a TF range tensor."""
  # Note: for static inputs (e.g. constants), tf.range errors out at graph
  # construction time, instead of returning an empty tensor. Preventing the
  # graph construction error aligns the semantics with Python.

  # TODO(mdan): We should optimize this when a full tensor is not required.
  if step is not UNSPECIFIED:
    # TODO(mdan): Add argument coercion similar to other cases.
    return math_ops.range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    stop = math_ops.maximum(start_or_stop, stop)
    return math_ops.range(start_or_stop, stop)
  start_or_stop = math_ops.maximum(start_or_stop, 0)
  return math_ops.range(start_or_stop)


def _py_range(start_or_stop, stop, step):
  if step is not UNSPECIFIED:
    return range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    return range(start_or_stop, stop)
  return range(start_or_stop)


def enumerate_(s, start=0):
  enumerate_override = registry_lookup(enumerate_registry, s)
  if enumerate_override is not None:
    return enumerate_override(s, start)
  return _py_enumerate(s, start)


def _py_enumerate(s, start=0):
  return enumerate(s, start)


def zip_(*iterables):
  zip_fn = _py_zip
  # If the overridden function is not the same across all iterables, use _py_zip
  for x in iterables:
    zip_override = registry_lookup(zip_registry, x)
    if zip_override is None or (zip_fn != _py_zip and zip_override != zip_fn):  # pylint: disable=comparison-with-callable
      zip_fn = _py_zip
      break
    zip_fn = zip_override
  return zip_fn(*iterables)


def _py_zip(*iterables):
  return zip(*iterables)


def map_(fn, *iterables):
  map_fn = _py_map
  # If the overridden function is not the same across all iterables, use _py_map
  for x in iterables:
    map_override = registry_lookup(map_registry, x)
    if map_override is None or (map_fn != _py_map and map_override != map_fn):  # pylint: disable=comparison-with-callable
      map_fn = _py_map
      break
    map_fn = map_override
  return map_fn(fn, *iterables)


def _py_map(fn, *iterables):
  return map(fn, *iterables)


def next_(iterator, default=UNSPECIFIED):
  next_override = registry_lookup(next_registry, iterator)
  if next_override is not None:
    return next_override(iterator, default)
  return next_py(iterator, default)


def next_py(iterator, default=UNSPECIFIED):
  if default is UNSPECIFIED:
    return next(iterator)
  return next(iterator, default)


def filter_(function, iterable):
  filter_override = registry_lookup(filter_registry, iterable)
  if filter_override is not None:
    return filter_override(function, iterable)
  return _py_filter(function, iterable)


def _py_filter(function, iterable):
  return filter(function, iterable)


def any_(iterable):
  any_override = registry_lookup(any_registry, iterable)
  if any_override is not None:
    return any_override(iterable)
  return _py_any(iterable)


def _py_any(iterable):
  return any(iterable)


def all_(iterable):
  all_override = registry_lookup(all_registry, iterable)
  if all_override is not None:
    return all_override(iterable)
  return _py_all(iterable)


def _py_all(iterable):
  return all(iterable)


def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
  sorted_override = registry_lookup(sorted_registry, iterable)
  if sorted_override is not None:
    return sorted_override(iterable, key, reverse)
  return _py_sorted(iterable, key, reverse)


def _py_sorted(iterable, key, reverse):
  if key is not UNSPECIFIED and reverse is UNSPECIFIED:
    return sorted(iterable, key=key)
  if key is UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, reverse=reverse)
  if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, key=key, reverse=reverse)
  return sorted(iterable)


SUPPORTED_BUILTINS = (abs, float, int, len, print, range, enumerate, zip, map,
                      filter, any, all, sorted)

BUILTIN_FUNCTIONS_MAP = {
    'abs': abs_,
    'any': any_,
    'all': all_,
    'enumerate': enumerate_,
    'filter': filter_,
    'float': float_,
    'int': int_,
    'len': len_,
    'map': map_,
    'next': next_,
    'print': print_,
    'range': range_,
    'sorted': sorted_,
    'zip': zip_,
}
