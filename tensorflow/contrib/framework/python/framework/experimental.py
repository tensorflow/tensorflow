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

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils


def _add_experimental_function_notice_to_docstring(doc):
  """Adds an experimental notice to a docstring for experimental functions."""
  return decorator_utils.add_notice_to_docstring(
      doc, '',
      'EXPERIMENTAL FUNCTION',
      '(experimental)', ['THIS FUNCTION IS EXPERIMENTAL. It may change or '
                         'be removed at any time, and without warning.'])


def experimental(func):
  """Decorator for marking functions or methods experimental.

  This decorator logs an experimental warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is experimental and may change or be removed at
    any time, and without warning.

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (experimental)' is appended
  to the first line of the docstring and a notice is prepended to the rest of
  the docstring.

  Args:
    func: A function or method to mark experimental.

  Returns:
    Decorated function or method.
  """
  decorator_utils.validate_callable(func, 'experimental')
  @functools.wraps(func)
  def new_func(*args, **kwargs):
    logging.warning(
        '%s (from %s) is experimental and may change or be removed at '
        'any time, and without warning.',
        decorator_utils.get_qualified_name(func), func.__module__)
    return func(*args, **kwargs)
  new_func.__doc__ = _add_experimental_function_notice_to_docstring(
      func.__doc__)
  return new_func
