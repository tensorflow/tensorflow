# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Classes used to handle thread-local stacks."""

import threading

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


class DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super().__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if self.stack else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    self.stack.append(default)
    try:
      yield default
    finally:
      # stack may be empty if reset() was called
      if self.stack:
        if self._enforce_nesting:
          if self.stack[-1] is not default:
            raise AssertionError(
                "Nesting violated for default stack of %s objects" %
                type(default))
          self.stack.pop()
        else:
          self.stack.remove(default)


_default_session_stack = DefaultStack()


def default_session(session):
  """Python "with" handler for defining a default session.

  This function provides a means of registering a session for handling
  Tensor.eval() and Operation.run() calls. It is primarily intended for use
  by session.Session, but can be used with any object that implements
  the Session.run() interface.

  Use with the "with" keyword to specify that Tensor.eval() and Operation.run()
  invocations within the scope of a block should be executed by a particular
  session.

  The default session applies to the current thread only, so it is always
  possible to inspect the call stack and determine the scope of a default
  session. If you create a new thread, and wish to use the default session
  in that thread, you must explicitly add a "with ops.default_session(sess):"
  block in that thread's function.

  Example:
    The following code examples are equivalent:

    # 1. Using the Session object directly:
    sess = ...
    c = tf.constant(5.0)
    sess.run(c)

    # 2. Using default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      result = c.eval()

    # 3. Overriding default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      with ops.default_session(...):
        c.eval(session=sess)

  Args:
    session: The session to be installed as the default session.

  Returns:
    A context manager for the default session.
  """
  return _default_session_stack.get_controller(session)


@tf_export(v1=["get_default_session"])
def get_default_session():
  """Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
  return _default_session_stack.get_default()
