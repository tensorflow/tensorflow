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

import inspect
import numbers
import os
import re

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_export
from tensorflow.python.types import core
from tensorflow.python.util import nest


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


def _np_doc_helper(f, np_f, np_fun_name=None, unsupported_params=None,
                   link=None):
  """Helper to get docs."""
  assert np_f or np_fun_name
  if not np_fun_name:
    np_fun_name = np_f.__name__
  doc = 'TensorFlow variant of NumPy\'s `%s`.\n\n' % np_fun_name
  if unsupported_params:
    doc += 'Unsupported arguments: ' + ', '.join(
        '`' + name + '`' for name in unsupported_params) + '.\n\n'
  if _has_docstring(f):
    doc += f.__doc__
    doc = _add_blank_line(doc)
  # TODO(wangpeng): Re-enable the following and choose inlined vs. link to numpy
  #   doc according to some global switch.
  doc = _add_np_doc(doc, np_fun_name, np_f, link=link)
  return doc


_np_doc_form = os.getenv('TF_NP_DOC_FORM', 'stable')


def get_np_doc_form():
  """Gets the form of the original numpy docstrings.

  Returns:
    See `set_np_doc_form` for the list of valid values.
  """
  return _np_doc_form


def set_np_doc_form(value):
  r"""Selects the form of the original numpy docstrings.

  This function sets a global variable that controls how a tf-numpy symbol's
  docstring should refer to the original numpy docstring. If `value` is
  `'inlined'`, the numpy docstring will be verbatim copied into the tf-numpy
  docstring. Otherwise, a link to the original numpy docstring will be
  added. Which numpy version the link points to depends on `value`:
  * `'stable'`: the current stable version;
  * `'dev'`: the current development version;
  * pattern `\d+(\.\d+(\.\d+)?)?`: `value` will be treated as a version number,
    e.g. '1.16'.

  Args:
    value: the value to set the global variable to.
  """
  global _np_doc_form
  _np_doc_form = value


class Link:

  def __init__(self, v):
    self.value = v


class AliasOf:

  def __init__(self, v):
    self.value = v


class NoLink:
  pass


def generate_link(flag, np_fun_name):
  """Generates link from numpy function name.

  Args:
    flag: the flag to control link form. See `set_np_doc_form`.
    np_fun_name: the numpy function name.

  Returns:
    A string.
  """
  # Only adds link in this case
  if flag == 'dev':
    template = 'https://numpy.org/devdocs/reference/generated/numpy.%s.html'
  elif flag == 'stable':
    template = (
        'https://numpy.org/doc/stable/reference/generated/numpy.%s.html')
  elif re.match(r'\d+(\.\d+(\.\d+)?)?$', flag):
    # `flag` is the version number
    template = (f'https://numpy.org/doc/{flag}/reference/generated/numpy.%s.html')
  else:
    return None
  return template % np_fun_name


_is_check_link = (os.getenv('TF_NP_CHECK_LINK', 'False') in
                  ('True', 'true', '1'))


def is_check_link():
  return _is_check_link


def set_check_link(value):
  global _is_check_link
  _is_check_link = value


def _add_np_doc(doc, np_fun_name, np_f, link):
  """Appends the numpy docstring to `doc`, according to `set_np_doc_form`.

  See `set_np_doc_form` for how it controls the form of the numpy docstring.

  Args:
    doc: the docstring to be appended to.
    np_fun_name: the name of the numpy function.
    np_f: (optional) the numpy function.
    link: (optional) which link to use. See `np_doc` for details.

  Returns:
    `doc` with numpy docstring appended.
  """
  flag = get_np_doc_form()
  if flag == 'inlined':
    if _has_docstring(np_f):
      doc += 'Documentation for `numpy.%s`:\n\n' % np_fun_name
      # TODO(wangpeng): It looks like code snippets in numpy doc don't work
      # correctly with doctest. Fix that and remove the reformatting of the np_f
      # comment.
      doc += np_f.__doc__.replace('>>>', '>')
  elif isinstance(flag, str):
    if link is None:
      url = generate_link(flag, np_fun_name)
    elif isinstance(link, AliasOf):
      url = generate_link(flag, link.value)
    elif isinstance(link, Link):
      url = link.value
    else:
      url = None
    if url is not None:
      if is_check_link():
        # Imports locally because some builds may not have `requests`
        import requests  # pylint: disable=g-import-not-at-top
        r = requests.head(url)
        if r.status_code != 200:
          raise ValueError(
              f'Check link failed at [{url}] with status code {r.status_code}. '
              f'Argument `np_fun_name` is {np_fun_name}.')
      doc += 'See the NumPy documentation for [`numpy.%s`](%s).' % (
          np_fun_name, url)
  return doc


_is_sig_mismatch_an_error = (
    os.getenv('TF_NP_SIG_MISMATCH_IS_ERROR', 'False') in ('True', 'true', '1'))


def is_sig_mismatch_an_error():
  return _is_sig_mismatch_an_error


def set_is_sig_mismatch_an_error(value):
  global _is_sig_mismatch_an_error
  _is_sig_mismatch_an_error = value


def np_doc(np_fun_name, np_fun=None, export=True, unsupported_params=None,
           link=None):
  """Attachs numpy docstring to a function.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.
    export: whether to export this symbol under module
      `tf.experimental.numpy`. Note that if `export` is `True`, `np_fun` must be
      a function directly under the `numpy` module, not under any submodule of
      `numpy` (e.g. `numpy.random`).
    unsupported_params: (optional) the list of parameters not supported
      by tf.numpy.
    link: (optional) which link to use. If `None`, a default link generated from
      `np_fun_name` will be used. If an instance of `AliasOf`, `link.value` will
      be used in place of `np_fun_name` for the link generation. If an instance
      of `Link`, `link.value` will be used as the whole link. If an instance of
      `NoLink`, no link will be added.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name_orig, np_fun_orig = np_fun_name, np_fun
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)
  np_sig = _np_signature(np_fun)
  if unsupported_params is None:
    unsupported_params = []

  def decorator(f):
    """The decorator."""
    if hasattr(inspect, 'signature') and np_sig is not None:
      try:
        sig = inspect.signature(f)
      except ValueError:
        sig = None
      if sig is not None:
        for name, param in sig.parameters.items():
          np_param = np_sig.parameters.get(name)
          if np_param is None:
            if is_sig_mismatch_an_error():
              raise TypeError(
                  f'Cannot find parameter {name} in the numpy function\'s '
                  f'signature (which has these parameters: '
                  f'{list(np_sig.parameters.keys())}). Argument `np_fun_name` '
                  f'is {np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.')
            else:
              continue
          if (is_sig_mismatch_an_error() and
              not _is_compatible_param_kind(param.kind, np_param.kind)):
            raise TypeError(
                f'Parameter {name} is of kind {param.kind} while in numpy it '
                f'is of kind {np_param.kind}. Argument `np_fun_name` is '
                f'{np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.')
          has_default = (param.default != inspect.Parameter.empty)
          np_has_default = (np_param.default != inspect.Parameter.empty)
          if is_sig_mismatch_an_error() and has_default != np_has_default:
            raise TypeError(
                'Parameter {} should{} have a default value. Argument '
                '`np_fun_name` is {}. Argument `np_fun` is {}.'.format(
                    name, '' if np_has_default else ' not', np_fun_name_orig,
                    np_fun_orig))
        for name in np_sig.parameters:
          if name not in sig.parameters:
            unsupported_params.append(name)
    f.__doc__ = _np_doc_helper(
        f, np_fun, np_fun_name=np_fun_name,
        unsupported_params=unsupported_params, link=link)
    if export:
      return np_export.np_export(np_fun_name)(f)
    else:
      return f

  return decorator


def np_doc_only(np_fun_name, np_fun=None, export=True):
  """Attachs numpy docstring to a function.

  This differs from np_doc in that it doesn't check for a match in signature.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.
    export: whether to export this symbol under module
      `tf.experimental.numpy`. Note that if `export` is `True`, `np_f` must be a
      function directly under the `numpy` module, not under any submodule of
      `numpy` (e.g. `numpy.random`).

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)

  def decorator(f):
    f.__doc__ = _np_doc_helper(f, np_fun, np_fun_name=np_fun_name)
    if export:
      return np_export.np_export(np_fun_name)(f)
    else:
      return f

  return decorator


# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args
@np_doc('finfo')
def finfo(dtype):
  """Note that currently it just forwards to the numpy namesake, while
  tensorflow and numpy dtypes may have different properties."""
  return np.finfo(_to_numpy_type(dtype))
# pylint: enable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args


def _maybe_get_dtype(x):
  """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
  # Don't put np.ndarray in this list, because np.result_type looks at the
  # value (not just dtype) of np.ndarray to decide the result type.
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, indexed_slices.IndexedSlices) or tensor_util.is_tf_type(x):
    return _to_numpy_type(x.dtype)
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, (list, tuple)):
    raise ValueError(
        f'Cannot find dtype for type inference from argument `x` of a sequence '
        f'type {type(x)}. For sequences, please call this function on each '
        f'element individually.')
  return x


# Can't use np_doc because np.result_type is a builtin function.
@np_doc_only('result_type')
def result_type(*arrays_and_dtypes):  # pylint: disable=missing-function-docstring
  if ops.is_auto_dtype_conversion_enabled():
    # Use auto dtype conversion semantics for type inference.
    dtype, _ = flexible_dtypes.result_type(*arrays_and_dtypes)
    return dtype
  arrays_and_dtypes = [
      _maybe_get_dtype(x) for x in nest.flatten(arrays_and_dtypes)
  ]
  if not arrays_and_dtypes:
    # If arrays_and_dtypes is an empty list, let numpy decide what the dtype is.
    arrays_and_dtypes = [np.asarray([])]
  return np_dtypes._result_type(*arrays_and_dtypes)  # pylint: disable=protected-access


def result_type_unary(a, dtype):  # pylint: disable=missing-function-docstring
  """Find the result type from a single input and a dtype."""
  if dtype:
    # We need to let np_utils.result_type decide the dtype, not tf.zeros_like
    return result_type(dtype)

  # np_utils.result_type treats string inputs as dtype strings, not as strings.
  # but for unary we want to treat it as a string input.
  if isinstance(a, str):
    return np.unicode_
  elif isinstance(a, bytes):
    return np.bytes_

  # TF and numpy has different interpretations of Python types such as
  # `float`, so we let `np_utils.result_type` decide.
  return result_type(a)


def _result_type_binary(t1, t2):  # pylint: disable=missing-function-docstring
  """A specialization of result_type for 2 arguments for performance reasons."""
  try:
    return np_dtypes._result_type(_maybe_get_dtype(t1),  # pylint: disable=protected-access
                                  _maybe_get_dtype(t2))  # pylint: disable=protected-access
  except ValueError:
    return result_type(t1, t2)


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
    return tf_cond.cond(pred, true_fn, false_fn)
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
