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
"""Decorator that provides a warning if the wrapped object is never used."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import types

import six  # pylint: disable=unused-import

from tensorflow.python.util import tf_decorator
# pylint: enable=g-bad-import-order,g-import-not-at-top


# TODO(b/65412899): Re-implement to avoid leaking python objects.
# This function / class remains since the API is public (mark_used()).
def _add_should_use_warning(x, fatal_error=False):
  """Wraps object x so that if it is never used, a warning is logged.

  Args:
    x: Python object.
    fatal_error: Python bool.  If `True`, tf.logging.fatal is raised
      if the returned value is never used.

  Returns:
    An instance of `TFShouldUseWarningWrapper` which subclasses `type(x)`
    and is a very shallow wrapper for `x` which logs access into `x`.
  """
  del fatal_error
  if x is None:  # special corner case where x is None
    return x

  # TODO(apassos) we don't have an easier way to check because importing context
  # or ops here would create a BUILD dependency cycle.
  if type(x).__name__ == 'EagerTensor':
    return x

  def override_method(method):
    def fn(self, *args, **kwargs):
      return method(self, *args, **kwargs)
    return fn

  class TFShouldUseWarningWrapper(type(x)):
    """Wrapper for objects that keeps track of their use."""

    def __init__(self, true_self):
      self.__dict__ = true_self.__dict__

    # Not sure why this pylint warning is being used; this is not an
    # old class form.
    # pylint: disable=super-on-old-class
    def __getattribute__(self, name):
      return super(TFShouldUseWarningWrapper, self).__getattribute__(name)

    def mark_used(self, *args, **kwargs):
      return

    # pylint: enable=super-on-old-class

  for name in dir(TFShouldUseWarningWrapper):
    method = getattr(TFShouldUseWarningWrapper, name)
    if not isinstance(method, types.FunctionType):
      continue
    if name in ('__init__', '__getattribute__', '__del__', 'mark_used'):
      continue
    setattr(TFShouldUseWarningWrapper, name,
            functools.wraps(method)(override_method(method)))

  wrapped = TFShouldUseWarningWrapper(x)
  wrapped.__doc__ = x.__doc__  # functools.wraps fails on some objects.
  return wrapped


def should_use_result(fn):
  """Function wrapper that ensures the function's output is used.

  If the output is not used, a `tf.logging.error` is logged.

  An output is marked as used if any of its attributes are read, modified, or
  updated.  Examples when the output is a `Tensor` include:

  - Using it in any capacity (e.g. `y = t + 0`, `sess.run(t)`)
  - Accessing a property (e.g. getting `t.name` or `t.op`).

  Note, certain behaviors cannot be tracked - for these the object may not
  be marked as used.  Examples include:

  - `t != 0`.  In this case, comparison is done on types / ids.
  - `isinstance(t, tf.Tensor)`.  Similar to above.

  Args:
    fn: The function to wrap.

  Returns:
    The wrapped function.
  """
  def wrapped(*args, **kwargs):
    return _add_should_use_warning(fn(*args, **kwargs))
  return tf_decorator.make_decorator(
      fn, wrapped, 'should_use_result',
      ((fn.__doc__ or '') +
       ('\n\n  '
        '**NOTE** The output of this function should be used.  If it is not, '
        'a warning will be logged.  To mark the output as used, '
        'call its .mark_used() method.')))


def must_use_result_or_fatal(fn):
  """Function wrapper that ensures the function's output is used.

  If the output is not used, a `tf.logging.fatal` error is raised.

  An output is marked as used if any of its attributes are read, modified, or
  updated.  Examples when the output is a `Tensor` include:

  - Using it in any capacity (e.g. `y = t + 0`, `sess.run(t)`)
  - Accessing a property (e.g. getting `t.name` or `t.op`).

  Note, certain behaviors cannot be tracked - for these the object may not
  be marked as used.  Examples include:

  - `t != 0`.  In this case, comparison is done on types / ids.
  - `isinstance(t, tf.Tensor)`.  Similar to above.

  Args:
    fn: The function to wrap.

  Returns:
    The wrapped function.
  """
  def wrapped(*args, **kwargs):
    return _add_should_use_warning(fn(*args, **kwargs), fatal_error=True)
  return tf_decorator.make_decorator(
      fn, wrapped, 'must_use_result_or_fatal',
      ((fn.__doc__ or '') +
       ('\n\n  '
        '**NOTE** The output of this function must be used.  If it is not, '
        'a fatal error will be raised.  To mark the output as used, '
        'call its .mark_used() method.')))
