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

import copy
import sys
import traceback

import six  # pylint: disable=unused-import


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
# pylint: enable=g-bad-import-order,g-import-not-at-top


class _TFShouldUseHelper(object):
  """Object stored in TFShouldUse-wrapped objects.

  When it is deleted it will emit a warning or error if its `sate` method
  has not been called by time of deletion, and Tensorflow is not executing
  eagerly or inside a tf.function (which use autodeps and resolve the
  main issues this wrapper warns about).
  """

  def __init__(self, type_, repr_, stack_frame, error_in_function,
               warn_in_eager):
    self._type = type_
    self._repr = repr_
    self._stack_frame = stack_frame
    self._error_in_function = error_in_function
    if context.executing_eagerly():
      # If warn_in_eager, sated == False.  Otherwise true.
      self._sated = not warn_in_eager
    elif ops.get_default_graph()._building_function:  # pylint: disable=protected-access
      if error_in_function:
        self._sated = False
        ops.add_exit_callback_to_default_func_graph(
            lambda: self._check_sated(raise_error=True))
      else:
        self._sated = True
    else:
      # TF1 graph building mode
      self._sated = False

  def sate(self):
    self._sated = True
    self._type = None
    self._repr = None
    self._stack_frame = None
    self._logging_module = None

  def _check_sated(self, raise_error):
    """Check if the object has been sated."""
    if self._sated:
      return
    creation_stack = ''.join(
        [line.rstrip()
         for line in traceback.format_stack(self._stack_frame, limit=5)])
    if raise_error:
      try:
        raise RuntimeError(
            'Object was never used (type {}): {}.  If you want to mark it as '
            'used call its "mark_used()" method.  It was originally created '
            'here:\n{}'.format(self._type, self._repr, creation_stack))
      finally:
        self.sate()
    else:
      tf_logging.error(
          '==================================\n'
          'Object was never used (type {}):\n{}\nIf you want to mark it as '
          'used call its "mark_used()" method.\nIt was originally created '
          'here:\n{}\n'
          '=================================='
          .format(self._type, self._repr, creation_stack))

  def __del__(self):
    self._check_sated(raise_error=False)


def _new__init__(self, wrapped_value, tf_should_use_helper):
  # pylint: disable=protected-access
  self._tf_should_use_helper = tf_should_use_helper
  self._tf_should_use_wrapped_value = wrapped_value


def _new__setattr__(self, key, value):
  if key in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
    return object.__setattr__(self, key, value)
  return setattr(
      object.__getattribute__(self, '_tf_should_use_wrapped_value'),
      key, value)


def _new__getattribute__(self, key):
  if key not in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
    object.__getattribute__(self, '_tf_should_use_helper').sate()
  if key in ('_tf_should_use_helper', 'mark_used', '__setatt__'):
    return object.__getattribute__(self, key)
  return getattr(
      object.__getattribute__(self, '_tf_should_use_wrapped_value'), key)


def _new_mark_used(self, *args, **kwargs):
  object.__getattribute__(self, '_tf_should_use_helper').sate()
  try:
    mu = object.__getattribute__(
        object.__getattribute__(self, '_tf_should_use_wrapped_value'),
        'mark_used')
    return mu(*args, **kwargs)
  except AttributeError:
    pass


_WRAPPERS = {}


def _get_wrapper(x, tf_should_use_helper):
  """Create a wrapper for object x, whose class subclasses type(x).

  The wrapper will emit a warning if it is deleted without any of its
  properties being accessed or methods being called.

  Args:
    x: The instance to wrap.
    tf_should_use_helper: The object that tracks usage.

  Returns:
    An object wrapping `x`, of type `type(x)`.
  """
  type_x = type(x)
  memoized = _WRAPPERS.get(type_x, None)
  if memoized:
    return memoized(x, tf_should_use_helper)

  tx = copy.deepcopy(type_x)
  copy_tx = type(tx.__name__, tx.__bases__, dict(tx.__dict__))
  copy_tx.__init__ = _new__init__
  copy_tx.__getattribute__ = _new__getattribute__
  copy_tx.mark_used = _new_mark_used
  copy_tx.__setattr__ = _new__setattr__
  _WRAPPERS[type_x] = copy_tx

  return copy_tx(x, tf_should_use_helper)


def _add_should_use_warning(x, error_in_function=False, warn_in_eager=False):
  """Wraps object x so that if it is never used, a warning is logged.

  Args:
    x: Python object.
    error_in_function: Python bool.  If `True`, a `RuntimeError` is raised
      if the returned value is never used when created during `tf.function`
      tracing.
    warn_in_eager: Python bool. If `True` raise warning if in Eager mode as well
      as graph mode.

  Returns:
    An instance of `TFShouldUseWarningWrapper` which subclasses `type(x)`
    and is a very shallow wrapper for `x` which logs access into `x`.
  """
  if x is None or (isinstance(x, list) and not x):
    return x

  if context.executing_eagerly() and not warn_in_eager:
    return x

  if ops.get_default_graph()._building_function and not error_in_function:  # pylint: disable=protected-access
    # We don't currently log warnings in tf.function calls, so just skip it.
    return x

  # Extract the current frame for later use by traceback printing.
  try:
    raise ValueError()
  except ValueError:
    stack_frame = sys.exc_info()[2].tb_frame.f_back

  tf_should_use_helper = _TFShouldUseHelper(
      type_=type(x),
      repr_=repr(x),
      stack_frame=stack_frame,
      error_in_function=error_in_function,
      warn_in_eager=warn_in_eager)

  return _get_wrapper(x, tf_should_use_helper)


def should_use_result(fn=None, warn_in_eager=False, error_in_function=False):
  """Function wrapper that ensures the function's output is used.

  If the output is not used, a `logging.error` is logged.  If
  `error_in_function` is set, then a `RuntimeError` will be raised at the
  end of function tracing if the output is not used by that point.

  An output is marked as used if any of its attributes are read, modified, or
  updated.  Examples when the output is a `Tensor` include:

  - Using it in any capacity (e.g. `y = t + 0`, `sess.run(t)`)
  - Accessing a property (e.g. getting `t.name` or `t.op`).
  - Calling `t.mark_used()`.

  Note, certain behaviors cannot be tracked - for these the object may not
  be marked as used.  Examples include:

  - `t != 0`.  In this case, comparison is done on types / ids.
  - `isinstance(t, tf.Tensor)`.  Similar to above.

  Args:
    fn: The function to wrap.
    warn_in_eager: Whether to create warnings in Eager as well.
    error_in_function: Whether to raise an error when creating a tf.function.

  Returns:
    The wrapped function.
  """
  def decorated(fn):
    def wrapped(*args, **kwargs):
      return _add_should_use_warning(fn(*args, **kwargs),
                                     warn_in_eager=warn_in_eager,
                                     error_in_function=error_in_function)
    return tf_decorator.make_decorator(
        target=fn,
        decorator_func=wrapped,
        decorator_name='should_use_result',
        decorator_doc=(
            (fn.__doc__ or '') +
            ('\n\n  '
             '**NOTE** The output of this function should be used.  If it is '
             'not, a warning will be logged or an error may be raised.  '
             'To mark the output as used, call its .mark_used() method.')))

  if fn is not None:
    return decorated(fn)
  else:
    return decorated
