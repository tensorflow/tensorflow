# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to retrieve function args."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import six

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def _is_bounded_method(fn):
  _, fn = tf_decorator.unwrap(fn)
  return tf_inspect.ismethod(fn) and (fn.__self__ is not None)


def _is_callable_object(obj):
  return hasattr(obj, '__call__') and tf_inspect.ismethod(obj.__call__)


def fn_args(fn):
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  if isinstance(fn, functools.partial):
    args = fn_args(fn.func)
    args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
  else:
    if _is_callable_object(fn):
      fn = fn.__call__
    args = tf_inspect.getfullargspec(fn).args
    if _is_bounded_method(fn):
      args.remove('self')
  return tuple(args)


def get_func_name(func):
  """Returns name of passed callable."""
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func):
      return func.__name__
    elif tf_inspect.ismethod(func):
      return '%s.%s' % (six.get_method_self(func).__class__.__name__,
                        six.get_method_function(func).__name__)
    else:  # Probably a class instance with __call__
      return type(func)
  else:
    raise ValueError('Argument must be callable')


def get_func_code(func):
  """Returns func_code of passed callable."""
  _, func = tf_decorator.unwrap(func)
  if callable(func):
    if tf_inspect.isfunction(func) or tf_inspect.ismethod(func):
      return six.get_function_code(func)
    elif hasattr(func, '__call__'):
      return six.get_function_code(func.__call__)
    else:
      raise ValueError('Unhandled callable, type=%s' % type(func))
  else:
    raise ValueError('Argument must be callable')
