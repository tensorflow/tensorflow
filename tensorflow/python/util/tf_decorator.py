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
"""Base TFDecorator class and utility functions for working with decorators.

There are two ways to create decorators that TensorFlow can introspect into.
This is important for documentation generation purposes, so that function
signatures aren't obscured by the (*args, **kwds) signature that decorators
often provide.

1. Call `tf_decorator.make_decorator` on your wrapper function. If your
decorator is stateless, or can capture all of the variables it needs to work
with through lexical closure, this is the simplest option. Create your wrapper
function as usual, but instead of returning it, return
`tf_decorator.make_decorator(your_wrapper)`. This will attach some decorator
introspection metadata onto your wrapper and return it.

Example:

  def print_hello_before_calling(target):
    def wrapper(*args, **kwargs):
      print('hello')
      return target(*args, **kwargs)
    return tf_decorator.make_decorator(wrapper)

2. Derive from TFDecorator. If your decorator needs to be stateful, you can
implement it in terms of a TFDecorator. Store whatever state you need in your
derived class, and implement the `__call__` method to do your work before
calling into your target. You can retrieve the target via
`super(MyDecoratorClass, self).decorated_target`, and call it with whatever
parameters it needs.

Example:

  class CallCounter(tf_decorator.TFDecorator):
    def __init__(self, target):
      super(CallCounter, self).__init__('count_calls', target)
      self.call_count = 0

    def __call__(self, *args, **kwargs):
      self.call_count += 1
      return super(CallCounter, self).decorated_target(*args, **kwargs)

  def count_calls(target):
    return CallCounter(target)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools as _functools
import inspect as _inspect


def make_decorator(target,
                   decorator_func,
                   decorator_name=None,
                   decorator_doc='',
                   decorator_argspec=None):
  """Make a decorator from a wrapper and a target.

  Args:
    target: The final callable to be wrapped.
    decorator_func: The wrapper function.
    decorator_name: The name of the decorator. If `None`, the name of the
      function calling make_decorator.
    decorator_doc: Documentation specific to this application of
      `decorator_func` to `target`.
    decorator_argspec: The new callable signature of this decorator.

  Returns:
    The `decorator_func` argument with new metadata attached.
  """
  if decorator_name is None:
    decorator_name = _inspect.stack()[1][3]  # Caller's name.
  decorator = TFDecorator(decorator_name, target, decorator_doc,
                          decorator_argspec)
  setattr(decorator_func, '_tf_decorator', decorator)
  decorator_func.__name__ = target.__name__
  decorator_func.__doc__ = decorator.__doc__
  decorator_func.__wrapped__ = target
  return decorator_func


def unwrap(maybe_tf_decorator):
  """Unwraps an object into a list of TFDecorators and a final target.

  Args:
    maybe_tf_decorator: Any callable object.

  Returns:
    A tuple whose first element is an list of TFDecorator-derived objects that
    were applied to the final callable target, and whose second element is the
    final undecorated callable target. If the `maybe_tf_decorator` parameter is
    not decorated by any TFDecorators, the first tuple element will be an empty
    list. The `TFDecorator` list is ordered from outermost to innermost
    decorators.
  """
  decorators = []
  cur = maybe_tf_decorator
  while True:
    if isinstance(cur, TFDecorator):
      decorators.append(cur)
    elif hasattr(cur, '_tf_decorator'):
      decorators.append(getattr(cur, '_tf_decorator'))
    else:
      break
    cur = decorators[-1].decorated_target
  return decorators, cur


class TFDecorator(object):
  """Base class for all TensorFlow decorators.

  TFDecorator captures and exposes the wrapped target, and provides details
  about the current decorator.
  """

  def __init__(self,
               decorator_name,
               target,
               decorator_doc='',
               decorator_argspec=None):
    self._decorated_target = target
    self._decorator_name = decorator_name
    self._decorator_doc = decorator_doc
    self._decorator_argspec = decorator_argspec
    self.__name__ = target.__name__
    if self._decorator_doc:
      self.__doc__ = self._decorator_doc
    elif target.__doc__:
      self.__doc__ = target.__doc__
    else:
      self.__doc__ = ''

  def __get__(self, obj, objtype):
    return _functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs):
    return self._decorated_target(*args, **kwargs)

  @property
  def decorated_target(self):
    return self._decorated_target

  @property
  def decorator_name(self):
    return self._decorator_name

  @property
  def decorator_doc(self):
    return self._decorator_doc

  @property
  def decorator_argspec(self):
    return self._decorator_argspec
