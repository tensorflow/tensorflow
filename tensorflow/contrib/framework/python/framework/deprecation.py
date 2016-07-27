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


def _add_deprecation_to_docstring(doc, date, instructions):
  """Adds a deprecation notice to a docstring."""
  if not doc:
    lines = ['DEPRECATED FUNCTION']
  else:
    lines = doc.splitlines()
    lines[0] += ' (deprecated)'

  notice = [
      '',
      'THIS FUNCTION IS DEPRECATED. It will be removed after %s.' % date,
      'Instructions for updating:',
      '%s' % instructions,
  ]

  if len(lines) > 1:
    # Make sure that we keep our distance from the main body
    if lines[1].strip():
      notice += ['']

    lines = [lines[0]] + notice + lines[1:]
  else:
    lines += notice

  return '\n'.join(lines)


def deprecated(date, instructions):
  """Decorator for marking functions or methods deprecated.

  This decorator adds a deprecation warning to a function's docstring. It has
  the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  whenever the decorated function is called. <function> will include the class
  name if it is a method.

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
  if not date:
    raise ValueError('Tell us what date this will be deprecated!')
  if not re.match(r'20\d\d-[01]\d-[0123]\d', date):
    raise ValueError('Date must be YYYY-MM-DD.')
  if not instructions:
    raise ValueError('Don\'t deprecate things without conversion instructions!')

  def deprecated_wrapper(func):
    """Deprecation wrapper."""
    if not hasattr(func, '__call__'):
      raise ValueError(
          '%s is not a function.'
          'If this is a property, apply @deprecated after @property.' % func)
    def new_func(*args, **kwargs):
      logging.warning('%s (from %s) is deprecated and will be removed after %s.'
                      '\nInstructions for updating:\n%s',
                      _get_qualified_name(func), func.__module__,
                      date, instructions)
      return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = _add_deprecation_to_docstring(func.__doc__, date,
                                                     instructions)
    new_func.__dict__.update(func.__dict__)
    return new_func
  return deprecated_wrapper
