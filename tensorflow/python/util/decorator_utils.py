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

"""Utility functions for writing decorators (which modify docstrings)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


def get_qualified_name(function):
  # Python 3
  if hasattr(function, '__qualname__'):
    return function.__qualname__

  # Python 2
  if hasattr(function, 'im_class'):
    return function.im_class.__name__ + '.' + function.__name__
  return function.__name__


def _normalize_docstring(docstring):
  """Normalizes the docstring.

  Replaces tabs with spaces, removes leading and trailing blanks lines, and
  removes any indentation.

  Copied from PEP-257:
  https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

  Args:
    docstring: the docstring to normalize

  Returns:
    The normalized docstring
  """
  if not docstring:
    return ''
  # Convert tabs to spaces (following the normal Python rules)
  # and split into a list of lines:
  lines = docstring.expandtabs().splitlines()
  # Determine minimum indentation (first line doesn't count):
  # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
  indent = sys.maxsize
  for line in lines[1:]:
    stripped = line.lstrip()
    if stripped:
      indent = min(indent, len(line) - len(stripped))
  # Remove indentation (first line is special):
  trimmed = [lines[0].strip()]
  if indent < sys.maxsize:
    for line in lines[1:]:
      trimmed.append(line[indent:].rstrip())
  # Strip off trailing and leading blank lines:
  while trimmed and not trimmed[-1]:
    trimmed.pop()
  while trimmed and not trimmed[0]:
    trimmed.pop(0)
  # Return a single string:
  return '\n'.join(trimmed)


def add_notice_to_docstring(
    doc, instructions, no_doc_str, suffix_str, notice):
  """Adds a deprecation notice to a docstring.

  Args:
    doc: The original docstring.
    instructions: A string, describing how to fix the problem.
    no_doc_str: The default value to use for `doc` if `doc` is empty.
    suffix_str: Is added to the end of the first line.
    notice: A list of strings. The main notice warning body.

  Returns:
    A new docstring, with the notice attached.

  Raises:
    ValueError: If `notice` is empty.
  """
  if not doc:
    lines = [no_doc_str]
  else:
    lines = _normalize_docstring(doc).splitlines()
    lines[0] += ' ' + suffix_str

  if not notice:
    raise ValueError('The `notice` arg must not be empty.')

  notice[0] = 'Warning: ' + notice[0]
  notice = [''] + notice + ([instructions] if instructions else [])

  if len(lines) > 1:
    # Make sure that we keep our distance from the main body
    if lines[1].strip():
      notice.append('')

    lines[1:1] = notice
  else:
    lines += notice

  return '\n'.join(lines)


def validate_callable(func, decorator_name):
  if not hasattr(func, '__call__'):
    raise ValueError(
        '%s is not a function. If this is a property, make sure'
        ' @property appears before @%s in your source code:'
        '\n\n@property\n@%s\ndef method(...)' % (
            func, decorator_name, decorator_name))


class classproperty(object):  # pylint: disable=invalid-name
  """Class property decorator.

  Example usage:

  class MyClass(object):

    @classproperty
    def value(cls):
      return '123'

  > print MyClass.value
  123
  """

  def __init__(self, func):
    self._func = func

  def __get__(self, owner_self, owner_cls):
    return self._func(owner_cls)


class _CachedClassProperty(object):
  """Cached class property decorator.

  Transforms a class method into a property whose value is computed once
  and then cached as a normal attribute for the life of the class.  Example
  usage:

  >>> class MyClass(object):
  ...   @cached_classproperty
  ...   def value(cls):
  ...     print("Computing value")
  ...     return '<property of %s>' % cls.__name__
  >>> class MySubclass(MyClass):
  ...   pass
  >>> MyClass.value
  Computing value
  '<property of MyClass>'
  >>> MyClass.value  # uses cached value
  '<property of MyClass>'
  >>> MySubclass.value
  Computing value
  '<property of MySubclass>'

  This decorator is similar to `functools.cached_property`, but it adds a
  property to the class, not to individual instances.
  """

  def __init__(self, func):
    self._func = func
    self._cache = {}

  def __get__(self, obj, objtype):
    if objtype not in self._cache:
      self._cache[objtype] = self._func(objtype)
    return self._cache[objtype]

  def __set__(self, obj, value):
    raise AttributeError('property %s is read-only' % self._func.__name__)

  def __delete__(self, obj):
    raise AttributeError('property %s is read-only' % self._func.__name__)


def cached_classproperty(func):
  return _CachedClassProperty(func)


cached_classproperty.__doc__ = _CachedClassProperty.__doc__
