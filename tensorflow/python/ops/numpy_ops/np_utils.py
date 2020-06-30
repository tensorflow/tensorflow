# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for internal use."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest

tensor_to_ndarray = np_arrays.tensor_to_ndarray


def _canonicalize_axis(axis, rank):
  return _canonicalize_axes([axis], rank)[0]


def _canonicalize_axes(axes, rank):
  rank = _maybe_static(rank)

  if isinstance(rank, core.Tensor):
    canonicalizer = (
        lambda axis: cond(axis < 0, lambda: axis + rank, lambda: axis))
  else:
    canonicalizer = lambda axis: axis + rank if axis < 0 else axis

  return [canonicalizer(axis) for axis in axes]


def _supports_signature():
  return hasattr(inspect, 'signature')


def _to_tf_type(dtype):
  """Converts a native python or numpy type to TF DType.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A tensorflow `DType`.
  """
  return dtypes.as_dtype(dtype)


def _to_numpy_type(dtype):
  """Converts a native python or TF DType to numpy type.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A NumPy `dtype`.
  """
  if isinstance(dtype, dtypes.DType):
    return dtype.as_numpy_dtype
  return np.dtype(dtype)


def isscalar(val):
  """Returns whether `val` is a scalar value or scalar Tensor."""
  if isinstance(val, np_arrays.ndarray):
    val = val.data
  if isinstance(val, core.Tensor):
    ndims = val.shape.ndims
    if ndims is not None:
      return ndims == 0
    else:
      return math_ops.equal(array_ops.rank(val), 0)
  else:
    return np.isscalar(val)


def _has_docstring(f):
  return (f and hasattr(f, '__doc__') and isinstance(f.__doc__, str) and
          f.__doc__)


def _add_blank_line(s):
  if s.endswith('\n'):
    return s + '\n'
  else:
    return s + '\n\n'


def _np_signature(f):
  """An enhanced inspect.signature that can handle numpy.ufunc."""
  # TODO(wangpeng): consider migrating away from inspect.signature.
  # inspect.signature is supported in Python 3.3.
  if not hasattr(inspect, 'signature'):
    return None
  if f is None:
    return None
  if not isinstance(f, np.ufunc):
    try:
      return inspect.signature(f)
    except ValueError:
      return None

  def names_from_num(prefix, n):
    if n <= 0:
      return []
    elif n == 1:
      return [prefix]
    else:
      return [prefix + str(i + 1) for i in range(n)]

  input_names = names_from_num('x', f.nin)
  output_names = names_from_num('out', f.nout)
  keyword_only_params = [('where', True), ('casting', 'same_kind'),
                         ('order', 'K'), ('dtype', None), ('subok', True),
                         ('signature', None), ('extobj', None)]
  params = []
  params += [
      inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
      for name in input_names
  ]
  if f.nout > 1:
    params += [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_ONLY, default=None)
        for name in output_names
    ]
  params += [
      inspect.Parameter(
          'out',
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          default=None if f.nout == 1 else (None,) * f.nout)
  ]
  params += [
      inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
      for name, default in keyword_only_params
  ]
  return inspect.Signature(params)


# Python 2 doesn't allow keyword-only argument. Python prior to 3.8 doesn't
# allow positional-only argument. So we conflate positional-only, keyword-only
# and positional-or-keyword arguments here.
def _is_compatible_param_kind(a, b):

  def relax(k):
    if k in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.KEYWORD_ONLY):
      return inspect.Parameter.POSITIONAL_OR_KEYWORD
    return k

  return relax(a) == relax(b)


def _prepare_np_fun_name_and_fun(np_fun_name, np_fun):
  """Mutually propagates information between `np_fun_name` and `np_fun`.

  If one is None and the other is not, we'll try to make the former not None in
  a best effort.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: the numpy function whose docstring will be used.

  Returns:
    Processed `np_fun_name` and `np_fun`.
  """
  if np_fun_name is not None:
    assert isinstance(np_fun_name, str)
  if np_fun is not None:
    assert not isinstance(np_fun, str)
  if np_fun is None:
    assert np_fun_name is not None
    try:
      np_fun = getattr(np, str(np_fun_name))
    except AttributeError:
      np_fun = None
  if np_fun_name is None:
    assert np_fun is not None
    np_fun_name = np_fun.__name__
  return np_fun_name, np_fun


def _np_doc_helper(f, np_f, np_fun_name=None, unsupported_params=None):
  """Helper to get docs."""
  assert np_f or np_fun_name
  if not np_fun_name:
    np_fun_name = np_f.__name__
  doc = 'TensorFlow variant of `numpy.%s`.\n\n' % np_fun_name
  if unsupported_params:
    doc += 'Unsupported arguments: ' + ', '.join(
        '`' + name + '`' for name in unsupported_params) + '.\n\n'
  if _has_docstring(f):
    doc += f.__doc__
    doc = _add_blank_line(doc)
  if _has_docstring(np_f):
    doc += 'Documentation for `numpy.%s`:\n\n' % np_f.__name__
    # TODO(wangpeng): It looks like code snippets in numpy doc don't work
    # correctly with doctest. Fix that and remove the reformatting of the np_f
    # comment.
    doc += np_f.__doc__.replace('>>>', '>')
  return doc


def np_doc(np_fun_name, np_fun=None):
  """Attachs numpy docstring to a function.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)
  np_sig = _np_signature(np_fun)

  def decorator(f):
    """The decorator."""
    unsupported_params = []
    if hasattr(inspect, 'signature') and np_sig is not None:
      try:
        sig = inspect.signature(f)
      except ValueError:
        sig = None
      # TODO(wangpeng): Enable this.
      # Looks like this may not work with different versions of numpy.
      # if sig is not None:
      #   for name, param in sig.parameters.items():
      #     np_param = np_sig.parameters.get(name)
      #     if np_param is None:
      #       raise TypeError('Cannot find parameter "%s" in the numpy
      #          function\'s ' 'signature' % name)
      #     if not _is_compatible_param_kind(param.kind, np_param.kind):
      #       raise TypeError(
      #           'Parameter "%s" is of kind %s while in numpy it is of '
      #           'kind %s' % (name, param.kind, np_param.kind))
      #     has_default = (param.default != inspect.Parameter.empty)
      #     np_has_default = (np_param.default != inspect.Parameter.empty)
      #     if has_default != np_has_default:
      #       raise TypeError('Parameter "%s" should%s have a default value' %
      #                       (name, '' if np_has_default else ' not'))
      #   for name in np_sig.parameters:
      #     if name not in sig.parameters:
      #       unsupported_params.append(name)
    f.__doc__ = _np_doc_helper(
        f,
        np_fun,
        np_fun_name=np_fun_name,
        unsupported_params=unsupported_params)
    return f

  return decorator


def np_doc_only(np_fun_name, np_fun=None):
  """Attachs numpy docstring to a function.

  This differs from np_doc in that it doesn't check for a match in signature.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)

  def decorator(f):
    f.__doc__ = _np_doc_helper(f, np_fun, np_fun_name=np_fun_name)
    return f

  return decorator


# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args
@np_doc('finfo')
def finfo(dtype):
  """Note that currently it just forwards to the numpy namesake, while
  tensorflow and numpy dtypes may have different properties."""
  return np.finfo(_to_numpy_type(dtype))
# pylint: enable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args


# Can't use np_doc because np.result_type is a builtin function.
@np_doc_only('result_type')
def result_type(*arrays_and_dtypes):  # pylint: disable=missing-function-docstring
  def maybe_get_dtype(x):
    # Don't put np.ndarray in this list, because np.result_type looks at the
    # value (not just dtype) of np.ndarray to decide the result type.
    if isinstance(
        x, (np_arrays.ndarray, core.Tensor, indexed_slices.IndexedSlices)):
      return _to_numpy_type(x.dtype)
    elif isinstance(x, dtypes.DType):
      return _to_numpy_type(x)
    return x

  arrays_and_dtypes = [
      maybe_get_dtype(x) for x in nest.flatten(arrays_and_dtypes)
  ]
  if not arrays_and_dtypes:
    # If arrays_and_dtypes is an empty list, let numpy decide what the dtype is.
    arrays_and_dtypes = [np.asarray([])]
  return np_dtypes._result_type(*arrays_and_dtypes)  # pylint: disable=protected-access


@np_doc('promote_types')
def promote_types(type1, type2):  # pylint: disable=missing-function-docstring
  type1 = _to_numpy_type(type1)
  type2 = _to_numpy_type(type2)
  return np_dtypes.canonicalize_dtype(np.promote_types(type1, type2))


def tf_broadcast(*args):
  """Broadcast tensors.

  Args:
    *args: a list of tensors whose shapes are broadcastable against each other.

  Returns:
    Tensors broadcasted to the common shape.
  """
  if len(args) <= 1:
    return args
  sh = array_ops.shape(args[0])
  for arg in args[1:]:
    sh = array_ops.broadcast_dynamic_shape(sh, array_ops.shape(arg))
  return [array_ops.broadcast_to(arg, sh) for arg in args]


# TODO(wangpeng): Move the following functions to a separate file and check for
#   float dtypes in each of them.


def get_static_value(x):
  """A version of tf.get_static_value that returns None on float dtypes.

  It returns None on float dtypes in order to avoid breaking gradients.

  Args:
    x: a tensor.

  Returns:
    Same as `tf.get_static_value`, except that it returns None when `x` has a
    float dtype.
  """
  if isinstance(x, core.Tensor) and (x.dtype.is_floating or x.dtype.is_complex):
    return None
  return tensor_util.constant_value(x)


def _maybe_static(x):
  value = get_static_value(x)
  if value is None:
    return x
  else:
    return value


# All the following functions exist becaues get_static_value can't handle
# their TF counterparts.


def cond(pred, true_fn, false_fn):
  """A version of tf.cond that tries to evaluate the condition."""
  v = get_static_value(pred)
  if v is None:
    return control_flow_ops.cond(pred, true_fn, false_fn)
  if v:
    return true_fn()
  else:
    return false_fn()


def add(a, b):
  """A version of tf.add that eagerly evaluates if possible."""
  return _maybe_static(a) + _maybe_static(b)


def subtract(a, b):
  """A version of tf.subtract that eagerly evaluates if possible."""
  return _maybe_static(a) - _maybe_static(b)


def greater(a, b):
  """A version of tf.greater that eagerly evaluates if possible."""
  return _maybe_static(a) > _maybe_static(b)


def greater_equal(a, b):
  """A version of tf.greater_equal that eagerly evaluates if possible."""
  return _maybe_static(a) >= _maybe_static(b)


def less_equal(a, b):
  """A version of tf.less_equal that eagerly evaluates if possible."""
  return _maybe_static(a) <= _maybe_static(b)


def logical_and(a, b):
  """A version of tf.logical_and that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return _maybe_static(b)
      else:
        return a_value
    else:
      return a_value & _maybe_static(b)
  else:
    return a & _maybe_static(b)


def logical_or(a, b):
  """A version of tf.logical_or that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return a_value
      else:
        return _maybe_static(b)
    else:
      return a_value | _maybe_static(b)
  else:
    return a | _maybe_static(b)


def getitem(a, slice_spec):
  """A version of __getitem__ that eagerly evaluates if possible."""
  return _maybe_static(a)[slice_spec]


def reduce_all(input_tensor, axis=None, keepdims=False):
  """A version of tf.reduce_all that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_all(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.all(axis=axis, keepdims=keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False):
  """A version of tf.reduce_any that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_any(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.any(axis=axis, keepdims=keepdims)


def tf_rank(t):
  r = t.shape.rank
  if r is not None:
    return r
  return array_ops.rank(t)
