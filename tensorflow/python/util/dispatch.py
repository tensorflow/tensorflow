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
"""Type-based dispatch for TensorFlow ops.

"Operation dispatchers" can be used to override the behavior for TensorFlow ops
when they are called with otherwise unsupported argument types.  In particular,
when an operation is called with arguments that would cause it to raise a
TypeError, it falls back on its registered operation dispatchers.  If any
registered dispatchers can handle the arguments, then its result is returned.
Otherwise, the original TypeError is raised.

By default, dispatch support is added to the generated op wrappers for any
visible ops by default.  Ops that are implemented in Python can opt in to
dispatch support using the `add_dispatch_support` decorator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# Private function attribute used to store a list of dispatchers.
DISPATCH_ATTR = "_tf_dispatchers"


class OpDispatcher(object):
  """Abstract base class for TensorFlow operator dispatchers.

  Each operation dispatcher acts as an override handler for a single
  TensorFlow operation, and its results are used when the handler indicates
  that it can handle the operation's arguments (by returning any value other
  than `OpDispatcher.NOT_SUPPORTED`).
  """

  # Sentinel value that can be returned to indicate that an operation
  # dispatcher does not support a given set of arguments.
  NOT_SUPPORTED = object()

  def handle(self, args, kwargs):  # pylint: disable=unused-argument
    """Handle this dispatcher's operation with the specified arguments.

    If this operation dispatcher can handle the given arguments, then
    return an appropriate value (or raise an appropriate exception).

    Args:
      args: The arguments to the operation.
      kwargs: They keyword arguments to the operation.

    Returns:
      The result of the operation, or `OpDispatcher.NOT_SUPPORTED` if this
      dispatcher can not handle the given arguments.
    """
    return self.NOT_SUPPORTED

  def register(self, op):
    """Register this dispatcher as a handler for `op`.

    Args:
      op: Python function: the TensorFlow operation that should be handled. Must
        have a dispatch list (which is added automatically for generated ops,
        and can be added to Python ops using the `add_dispatch_support`
        decorator).
    """
    if not hasattr(op, DISPATCH_ATTR):
      raise AssertionError("Dispatching not enabled for %s" % op)
    getattr(op, DISPATCH_ATTR).append(self)


def dispatch(op, *args, **kwargs):
  """Returns the result from the first successful dispatcher for a given op.

  Calls the `handle` method of each `OpDispatcher` that has been registered
  to handle `op`, and returns the value from the first successful handler.

  Args:
    op: Python function: the operation to dispatch for.
    *args: The arguments to the operation.
    **kwargs: They keyword arguments to the operation.

  Returns:
    The result of the operation, or `NOT_SUPPORTED` if no registered
    dispatcher can handle the given arguments.
  """
  for dispatcher in getattr(op, DISPATCH_ATTR):
    result = dispatcher.handle(args, kwargs)
    if result is not OpDispatcher.NOT_SUPPORTED:
      return result
  return OpDispatcher.NOT_SUPPORTED


class _TypeBasedDispatcher(OpDispatcher):
  """Dispatcher that handles op if any arguments have a specified type.

  Checks the types of the arguments and keyword arguments (including elements
  of lists or tuples), and if any argument values have the indicated type(s),
  then delegates to an override function.
  """

  def __init__(self, override_func, types):
    self._types = types
    self._override_func = override_func

  def _handles(self, args, kwargs):
    for arg in itertools.chain(args, kwargs.values()):
      if (isinstance(arg, self._types) or
          (isinstance(arg, (list, tuple)) and
           any(isinstance(elt, self._types) for elt in arg))):
        return True
    return False

  def handle(self, args, kwargs):
    if self._handles(args, kwargs):
      return self._override_func(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED


# pylint: disable=g-doc-return-or-yield
def dispatch_for_types(op, *types):
  """Decorator to declare that a Python function overrides an op for a type.

  The decorated function is used to override `op` if any of the arguments or
  keyword arguments (including elements of lists or tuples) have one of the
  specified types.

  Example:

  ```python
  @dispatch_for_types(math_ops.add, RaggedTensor, RaggedTensorValue)
  def ragged_add(x, y, name=None): ...
  ```

  Args:
    op: Python function: the operation that should be overridden.
    *types: The argument types for which this function should be used.
  """

  def decorator(func):
    if tf_inspect.getargspec(func) != tf_inspect.getargspec(op):
      raise AssertionError("The decorated function's signature must exactly "
                           "match the signature of the overridden op.")
    _TypeBasedDispatcher(func, types).register(op)
    return func

  return decorator


# pylint: enable=g-doc-return-or-yield


def add_dispatch_list(target):
  """Decorator that adds a dispatch_list attribute to an op."""
  if hasattr(target, DISPATCH_ATTR):
    raise AssertionError("%s already has a dispatch list" % target)
  setattr(target, DISPATCH_ATTR, [])
  return target


def add_dispatch_support(target):
  """Decorator that adds a dispatch handling wrapper to an op."""
  def wrapper(*args, **kwargs):
    """Call target, and fall back on dispatchers if there is a TypeError."""
    try:
      return target(*args, **kwargs)
    except (TypeError, ValueError):
      # Note: convert_to_eager_tensor currently raises a ValueError, not a
      # TypeError, when given unexpected types.  So we need to catch both.
      result = dispatch(wrapper, *args, **kwargs)
      if result is not OpDispatcher.NOT_SUPPORTED:
        return result
      else:
        raise

  add_dispatch_list(wrapper)
  return tf_decorator.make_decorator(target, wrapper)
