# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tensor utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import re

from tensorflow.python.platform import tf_logging as logging


def _get_qualified_name(function):
  # Python 3
  if hasattr(function, '__qualname__'):
    return function.__qualname__

  # Python 2
  if hasattr(function, 'im_class'):
    return function.im_class.__name__ + '.' + function.__name__
  return function.__name__


def _add_deprecation_to_docstring(
    doc, instructions, no_doc_str, suffix_str, notice):
  """Adds a deprecation notice to a docstring."""
  if not doc:
    lines = [no_doc_str]
  else:
    lines = doc.splitlines()
    lines[0] += ' ' + suffix_str

  notice = [''] + notice + [instructions]

  if len(lines) > 1:
    # Make sure that we keep our distance from the main body
    if lines[1].strip():
      notice.append('')

    lines[1:1] = notice
  else:
    lines += notice

  return '\n'.join(lines)


def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated functions."""
  return _add_deprecation_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION',
      '(deprecated)', [
          'THIS FUNCTION IS DEPRECATED. It will be removed after %s.' % date,
          'Instructions for updating:'])


def _add_deprecated_arg_notice_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring for deprecated arguments."""
  return _add_deprecation_to_docstring(
      doc, instructions,
      'DEPRECATED FUNCTION ARGUMENTS',
      '(deprecated arguments)', [
          'SOME ARGUMENTS ARE DEPRECATED. '
          'They will be removed after %s.' % date,
          'Instructions for updating:'])


def _validate_deprecation_args(date, instructions):
  if not date:
    raise ValueError('Tell us what date this will be deprecated!')
  if not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError('Date must be YYYY-MM-DD.')
  if not instructions:
    raise ValueError('Don\'t deprecate things without conversion instructions!')


def _validate_callable(func, decorator_name):
  if not hasattr(func, '__call__'):
    raise ValueError(
        '%s is not a function. If this is a property, '
        'apply @%s after @property.' % (func, decorator_name))


def deprecated(date, instructions):
  """Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, or instructions are empty.
  """
  _validate_deprecation_args(date, instructions)

  def deprecated_wrapper(func):
    """Deprecation wrapper."""
    _validate_callable(func, 'deprecated')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      logging.warning(
          '%s (from %s) is deprecated and will be removed after %s.\n'
          'Instructions for updating:\n%s',
          _get_qualified_name(func), func.__module__, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_function_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper


def deprecated_args(date, instructions, *deprecated_arg_names):
  """Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names: String. The deprecated arguments.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, instructions are empty, or
      the deprecated arguments are not present in the function signature.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_arg_names:
    raise ValueError('Specify which argument is deprecated.')

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    _validate_callable(func, 'deprecated_args')

    arg_spec = inspect.getargspec(func)
    deprecated_positions = [
        (i, arg_name) for (i, arg_name) in enumerate(arg_spec.args)
        if arg_name in deprecated_arg_names]
    is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
    is_kwargs_deprecated = arg_spec.keywords in deprecated_arg_names

    if (len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated
        != len(deprecated_arg_names)):
      known_args = arg_spec.args + [arg_spec.varargs, arg_spec.keywords]
      missing_args = [arg_name for arg_name in deprecated_arg_names
                      if arg_name not in known_args]
      raise ValueError('The following deprecated arguments are not present '
                       'in the function signature: %s' % missing_args)

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      invalid_args = []
      for (i, arg_name) in deprecated_positions:
        if i < len(args):
          invalid_args.append(arg_name)
      if is_varargs_deprecated and len(args) > len(arg_spec.args):
        invalid_args.append(arg_spec.varargs)
      if is_kwargs_deprecated and kwargs:
        invalid_args.append(arg_spec.keywords)
      for arg_name in deprecated_arg_names:
        if arg_name in kwargs:
          invalid_args.append(arg_name)
      for arg_name in invalid_args:
        logging.warning(
            'Calling %s (from %s) with %s is deprecated and will be removed '
            'after %s.\nInstructions for updating:\n%s',
            _get_qualified_name(func), func.__module__,
            arg_name, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper


def deprecated_arg_values(date, instructions, **deprecated_kwargs):
  """Decorator for marking specific function argument values as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument values. It has the following format:

    Calling <function> (from <module>) with <arg>=<value> is deprecated and
    will be removed after <date>. Instructions for updating:
      <instructions>

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String. The date the function is scheduled to be removed. Must be
      ISO 8601 (YYYY-MM-DD).
    instructions: String. Instructions on how to update code using the
      deprecated function.
    **deprecated_kwargs: The deprecated argument values.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not in ISO 8601 format, or instructions are empty.
  """
  _validate_deprecation_args(date, instructions)
  if not deprecated_kwargs:
    raise ValueError('Specify which argument values are deprecated.')

  def deprecated_wrapper(func):
    """Deprecation decorator."""
    _validate_callable(func, 'deprecated_arg_values')
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      """Deprecation wrapper."""
      named_args = inspect.getcallargs(func, *args, **kwargs)
      for arg_name, arg_value in deprecated_kwargs.items():
        if arg_name in named_args and named_args[arg_name] == arg_value:
          logging.warning(
              'Calling %s (from %s) with %s=%s is deprecated and will be '
              'removed after %s.\nInstructions for updating:\n%s',
              _get_qualified_name(func), func.__module__,
              arg_name, arg_value, date, instructions)
      return func(*args, **kwargs)
    new_func.__doc__ = _add_deprecated_arg_notice_to_docstring(
        func.__doc__, date, instructions)
    return new_func
  return deprecated_wrapper
