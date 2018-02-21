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
"""Live entity inspection utilities.

This module contains whatever inspect doesn't offer out of the box.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.util import tf_inspect


def getcallargs(c, *args, **kwargs):
  """Extension of getcallargs to non-function callables."""
  if tf_inspect.isfunction(c):
    # The traditional getcallargs
    return tf_inspect.getcallargs(c, *args, **kwargs)

  if tf_inspect.isclass(c):
    # Constructors: pass a fake None for self, then remove it.
    arg_map = tf_inspect.getcallargs(c.__init__, None, *args, **kwargs)
    assert 'self' in arg_map, 'no "self" argument, is this not a constructor?'
    del arg_map['self']
    return arg_map

  if hasattr(c, '__call__'):
    # Callable objects: map self to the object itself
    return tf_inspect.getcallargs(c.__call__, *args, **kwargs)

  raise NotImplementedError('unknown callable "%s"' % type(c))


def getmethodclass(m, namespace):
  """Resolves a function's owner, e.g. a method's class."""

  # Instance method and class methods: should be bound to a non-null "self".
  # If self is a class, then it's a class method.
  if hasattr(m, '__self__'):
    if m.__self__:
      if tf_inspect.isclass(m.__self__):
        return m.__self__
      return type(m.__self__)

  # Class and static methods: platform specific.
  if hasattr(m, 'im_class'):  # Python 2
    return m.im_class

  if hasattr(m, '__qualname__'):  # Python 3
    qn = m.__qualname__.split('.')
    if len(qn) < 2:
      return None
    owner_name, func_name = qn[-2:]
    assert func_name == m.__name__, (
        'inconsistent names detected '
        '(__qualname__[1] = "%s", __name__ = "%s") for %s.' % (func_name,
                                                               m.__name__, m))
    if owner_name == '<locals>':
      return None
    if owner_name not in namespace:
      raise ValueError(
          'Could not resolve name "%s" while analyzing %s. Namespace:\n%s' %
          (owner_name, m, namespace))
    return namespace[owner_name]

  if six.PY2:
    # In Python 2 it's impossible, to our knowledge, to detect the class of a
    # static function. So we're forced to walk all the objects in the
    # namespace and see if they own it. If any reader finds a better solution,
    # please let us know.
    for _, v in namespace.items():
      if hasattr(v, m.__name__) and getattr(v, m.__name__) is m:
        return v

  return None
