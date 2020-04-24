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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

import numpy as np
import six

from tensorflow.python.autograph.utils import py_func
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest


# TODO(b/145618471): Remove this dependency.
# Lazy import to work around circular dependencies
input_lib = lazy_loader.LazyLoader(
    'input_lib', globals(),
    'tensorflow.python.distribute.input_lib')
parallel_ops = lazy_loader.LazyLoader(
    'parallel_ops', globals(),
    'tensorflow.python.ops.parallel_for.control_flow_ops')


UNSPECIFIED = object()


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

  # Python 2 doesn't support implicit argument super variants.
  if six.PY2:
    return f(*args)

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
  if tensor_util.is_tensor(x):
    return _tf_abs(x)
  if isinstance(x, dataset_ops.DatasetV2):
    return _tf_dataset_abs(x)
  return _py_abs(x)


def _tf_abs(x):
  return math_ops.abs(x)


def _tf_dataset_abs(x):
  specs = nest.flatten(x.element_spec)
  if len(specs) == 1:
    return x.map(math_ops.abs)
  return x.map(lambda *e: nest.map_structure(math_ops.abs, e))


def _py_abs(x):
  return abs(x)


def float_(x=0):
  if tensor_util.is_tensor(x):
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
  if tensor_util.is_tensor(x):
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
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tensor(s):
    return _tf_tensor_len(s)
  if isinstance(s, dataset_ops.DatasetV2):
    return _tf_dataset_len(s)
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
    with ops.control_dependencies([control_flow_ops.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)

  return control_flow_ops.cond(rank > 0, lambda: array_ops.shape(s)[0],
                               raise_zero_rank_error)


def _tf_dataset_len(s):
  l = cardinality.cardinality(s)
  msg = gen_string_ops.string_join([
      'len requires dataset with definitive cardinality, got ',
      gen_string_ops.as_string(l)
  ])
  # TODO (yongtang): UNKNOWN is treated as an error.
  # In case there are more UNKNOWN cases for dataset, we could
  # use dataset.reduce() to find out the length (in an expensive way).
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.logical_and(
              math_ops.not_equal(l, cardinality.INFINITE),
              math_ops.not_equal(l, cardinality.UNKNOWN)), [msg])
  ]):
    l = array_ops.identity(l)

  return l


def _py_len(s):
  return len(s)


def print_(*objects, **kwargs):
  """Overload of the print builtin."""
  # Note: Python 2.6 doesn't support explicit keywords after starargs.
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))

  # TODO(mdan): Use next.flatten(objects) instead?
  if any(tensor_util.is_tensor(o) for o in objects):
    # TODO(mdan): use tf.print instead.
    return _tf_py_func_print(objects, kwargs)
  else:
    _py_print(*objects, **kwargs)


def _py_print(*objects, **kwargs):
  print(*objects, **kwargs)


def _tf_py_func_print(objects, kwargs):
  """Overload of print_ as a py_func implementation."""
  override_kwargs = {k: v for k, v in kwargs.items() if v is not UNSPECIFIED}
  if 'flush' not in override_kwargs:
    # Defaulting to flushing the console in graph mode, which helps reduce
    # garbled output in IPython.
    override_kwargs['flush'] = True

  def print_wrapper(*vals):
    vals = tuple(v.numpy() if tensor_util.is_tensor(v) else v for v in vals)
    if not six.PY2:
      # TensorFlow doesn't seem to generate Unicode when passing strings to
      # py_func. This causes the print to add a "b'" wrapper to the output,
      # which is probably never what you want.
      vals = tuple(
          v.decode('utf-8') if isinstance(v, bytes) else v for v in vals)
    six.print_(*vals, **override_kwargs)

  return py_func.wrap_py_func(
      print_wrapper, None, objects, use_dummy_return=True)


def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
  if any(tensor_util.is_tensor(s) for s in (start_or_stop, stop, step)):
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
  if isinstance(s, dataset_ops.DatasetV2):
    return _tf_dataset_enumerate(s, start)
  if isinstance(
      s, (input_lib.DistributedIterator, input_lib.DistributedDataset)):
    raise NotImplementedError(
        'use a for loop over the dataset and keep a separate counter')
  return _py_enumerate(s, start)


def _tf_dataset_enumerate(s, start=0):
  return s.enumerate(start)


def _py_enumerate(s, start=0):
  return enumerate(s, start)


def zip_(*iterables):
  if all(isinstance(x, dataset_ops.DatasetV2) for x in iterables):
    return _tf_dataset_zip(*iterables)
  return _py_zip(*iterables)


def _tf_dataset_zip(*iterables):
  return dataset_ops.DatasetV2.zip(iterables)


def _py_zip(*iterables):
  return zip(*iterables)


def map_(fn, *iterables):
  if all(isinstance(x, dataset_ops.DatasetV2) for x in iterables):
    return _tf_dataset_map(fn, *iterables)
  return _py_map(fn, *iterables)


def _tf_dataset_map(fn, *iterables):
  return dataset_ops.DatasetV2.zip(iterables).map(fn)


def _py_map(fn, *iterables):
  return map(fn, *iterables)


def next_(iterator, default=UNSPECIFIED):
  if isinstance(iterator, iterator_ops.OwnedIterator):
    return next_tf_iterator(iterator, default)
  return next_py(iterator, default)


# TODO(mdan): These checks should be easier. Fix the nest API.
def _verify_spec_compatible(input_name, spec_name, input_, spec):
  """Verifies that a symbol has a type compatible vith a given spec.

  Here, compatibility is viewed in the general TensorFlow sense: that the dtypes
  are the same after implicit conversion, if both are tensors.

  This verifier ensures consistent treatment of types across AutoGraph.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify.
    spec: TypeSpec that `input_` must be compatible with.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  assert isinstance(spec, tensor_spec.TensorSpec)
  if input is None:
    # TODO(mdan): raise from None when switching to Py3.
    raise ValueError('{} cannot be None'.format(input_name))

  # TODO(mdan): Use TensorCompatible when ready.
  if isinstance(input_, (bool, int, float, str, np.ndarray)):
    input_ = ops.convert_to_tensor_v2(input_)

  input_dtype = getattr(input_, 'dtype', None)

  if input_dtype != spec.dtype:
    input_dtype_str = 'no dtype' if input_dtype is None else str(input_dtype)

    raise TypeError(
        '{} must have the same dtype as {}. Expected {}, got {}'.format(
            input_name, spec_name, spec.dtype, input_dtype_str))


def _verify_structure_compatible(input_name, spec_name, input_, spec):
  """Verifies that possibly-structured symbol has types compatible vith another.

  See _verify_spec_compatible for a more concrete meaning of "compatible".
  Unspec _verify_spec_compatible, which handles singular Tensor-spec objects,
  verify_structures_compatible can process structures recognized by tf.nest.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify. May, but doesn't need to, be a structure.
    spec: Any, value that `input_` must be compatible with. May, but doesn't
        need to, be a structure.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
  try:
    nest.assert_same_structure(input_, spec, expand_composites=True)
  except (ValueError, TypeError) as e:
    raise TypeError(
        '{} must have the same element structure as {}.\n\n{}'.format(
            input_name, spec_name, str(e)))

  nest.map_structure(
      functools.partial(_verify_spec_compatible, input_name, spec_name), input_,
      spec)


def next_tf_iterator(iterator, default=UNSPECIFIED):
  if default is UNSPECIFIED:
    # Without a default, fall back to the "normal" behavior which raises
    # a runtime exception.
    return next(iterator)
  opt_iterate = iterator_ops.get_next_as_optional(iterator)
  _verify_structure_compatible(
      'the default argument', 'the iterate', default, iterator.element_spec)
  return control_flow_ops.cond(
      opt_iterate.has_value(), opt_iterate.get_value, lambda: default)


def next_py(iterator, default=UNSPECIFIED):
  if default is UNSPECIFIED:
    return next(iterator)
  return next(iterator, default)


def filter_(function, iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_filter(function, iterable)
  return _py_filter(function, iterable)


def _tf_dataset_filter(function, iterable):
  return iterable.filter(function)


def _py_filter(function, iterable):
  return filter(function, iterable)


def any_(iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_any(iterable)
  return _py_any(iterable)


# any() operation is essentially a "if first True element exist".
# For that it could be translated to `filter(True)` to filter out
# only `True` element, and then `take(1)`. This works in tf.data
# as tf.data's filter+take is done in pipeline so it will stop
# as soon as `take(1)` returns.
def _tf_dataset_any(iterable):
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "any" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: x)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
  return ds


def _py_any(iterable):
  return any(iterable)


def all_(iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_all(iterable)
  return _py_all(iterable)


# all() operation is similar to any() and could be translated
# to `filter(False)` then `take(1)`, and check if `False` exists.
def _tf_dataset_all(iterable):
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "all" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: math_ops.logical_not(x))
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(True, dtype=dtypes.bool), lambda _, y: y)
  return ds


def _py_all(iterable):
  return all(iterable)


def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
  if tensor_util.is_tensor(iterable):
    return _tf_sorted(iterable, key, reverse)
  return _py_sorted(iterable, key, reverse)


def _tf_sorted(iterable, key, reverse):
  """Overload of sorted_ for Tensor iterable."""
  if reverse is UNSPECIFIED:
    direction = 'ASCENDING'
  else:
    direction = 'DESCENDING'
  if key is not UNSPECIFIED:
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

if six.PY2:
  SUPPORTED_BUILTINS += (xrange,)

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
    'xrange': range_,
    'zip': zip_,
}
