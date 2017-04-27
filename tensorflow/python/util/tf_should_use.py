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
import traceback
import types

from tensorflow.python.platform import tf_logging


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
  def override_method(method):
    def fn(self, *args, **kwargs):
      self._tf_object_has_been_used = True  # pylint: disable=protected-access
      return method(self, *args, **kwargs)
    return fn

  class TFShouldUseWarningWrapper(type(x)):
    """Wrapper for objects that keeps track of their use."""

    def __init__(self, true_self):
      self.__dict__ = true_self.__dict__
      stack = [x.strip() for x in traceback.format_stack()]
      # Remove top three stack entries from adding the wrapper
      self._tf_object_creation_stack = '\n'.join(stack[:-3])
      self._tf_object_has_been_used = False

    # Not sure why this pylint warning is being used; this is not an
    # old class form.
    # pylint: disable=super-on-old-class
    def __getattribute__(self, name):
      if name != '_tf_object_has_been_used':
        self._tf_object_has_been_used = True
      return super(TFShouldUseWarningWrapper, self).__getattribute__(name)

    def __del__(self):
      if not self._tf_object_has_been_used:
        if fatal_error:
          logger = tf_logging.fatal
        else:
          logger = tf_logging.error
        logger(
            'Object was never used: %s.\nIt was originally created here:\n%s'
            % (self, self._tf_object_creation_stack))

      if hasattr(super(TFShouldUseWarningWrapper, self), '__del__'):
        return super(TFShouldUseWarningWrapper, self).__del__()
    # pylint: enable=super-on-old-class

  for name in dir(TFShouldUseWarningWrapper):
    method = getattr(TFShouldUseWarningWrapper, name)
    if not isinstance(method, types.FunctionType):
      continue
    if name in ('__init__', '__getattribute__', '__del__'):
      continue
    setattr(TFShouldUseWarningWrapper, name,
            functools.wraps(method)(override_method(method)))

  wrapped = TFShouldUseWarningWrapper(x)
  wrapped.__doc__ = x.__doc__  # functools.wraps fails on some objects.
  wrapped._tf_object_has_been_used = False   # pylint: disable=protected-access
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
  return functools.wraps(fn)(wrapped)


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
  return functools.wraps(fn)(wrapped)
