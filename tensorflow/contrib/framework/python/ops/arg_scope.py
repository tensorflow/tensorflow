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
"""Contains the arg_scope used for scoping layers arguments.

  Allows one to define models much more compactly by eliminating boilerplate
  code. This is accomplished through the use of argument scoping (arg_scope).

  Example of how to use tf.contrib.framework.arg_scope:

  ```
  from third_party.tensorflow.contrib.layers.python import layers

  arg_scope = tf.contrib.framework.arg_scope

  with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)):
    net = layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')
  ```
  The first call to conv2d will behave as follows:
    layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                  initializer=layers.variance_scaling_initializer(),
                  regularizer=layers.l2_regularizer(0.05), scope='conv1')

  The second call to conv2d will also use the arg_scope's default for padding:
    layers.conv2d(inputs, 256, [5, 5], padding='SAME',
                  initializer=layers.variance_scaling_initializer(),
                  regularizer=layers.l2_regularizer(0.05), scope='conv2')

  Example of how to reuse an arg_scope:

  ```
  with arg_scope([layers.conv2d], padding='SAME',
                 initializer=layers.variance_scaling_initializer(),
                 regularizer=layers.l2_regularizer(0.05)) as sc:
    net = layers.conv2d(net, 256, [5, 5], scope='conv1')
    ....

  with arg_scope(sc):
    net = layers.conv2d(net, 256, [5, 5], scope='conv2')
  ```

  Example of how to use tf.contrib.framework.add_arg_scope to enable your
  function to be called within an arg_scope later:

  @tf.contrib.framework.add_arg_scope
  def conv2d(*args, **kwargs)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator

__all__ = [
    'arg_scope', 'add_arg_scope', 'current_arg_scope', 'has_arg_scope',
    'arg_scoped_arguments', 'arg_scope_func_key'
]

_ARGSTACK = [{}]

_DECORATED_OPS = {}


def _get_arg_stack():
  if _ARGSTACK:
    return _ARGSTACK
  else:
    _ARGSTACK.append({})
    return _ARGSTACK


def current_arg_scope():
  stack = _get_arg_stack()
  return stack[-1]


def arg_scope_func_key(op):
  return getattr(op, '_key_op', str(op))


def _name_op(op):
  return (op.__module__, op.__name__)


def _kwarg_names(func):
  kwargs_length = len(func.__defaults__) if func.__defaults__ else 0
  return func.__code__.co_varnames[-kwargs_length:func.__code__.co_argcount]


def _add_op(op):
  key_op = _key_op(op)
  _DECORATED_OPS[key_op] = _kwarg_names(op)


@tf_contextlib.contextmanager
def arg_scope(list_ops_or_scope, **kwargs):
  """Stores the default arguments for the given set of list_ops.

  For usage, please see examples at top of the file.

  Args:
    list_ops_or_scope: List or tuple of operations to set argument scope for or
      a dictionary containing the current scope. When list_ops_or_scope is a
      dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
      then every op in it need to be decorated with @add_arg_scope to work.
    **kwargs: keyword=value that will define the defaults for each op in
              list_ops. All the ops need to accept the given set of arguments.

  Yields:
    the current_scope, which is a dictionary of {op: {arg: value}}
  Raises:
    TypeError: if list_ops is not a list or a tuple.
    ValueError: if any op in list_ops has not be decorated with @add_arg_scope.
  """
  if isinstance(list_ops_or_scope, dict):
    # Assumes that list_ops_or_scope is a scope that is being reused.
    if kwargs:
      raise ValueError('When attempting to re-use a scope by suppling a'
                       'dictionary, kwargs must be empty.')
    current_scope = list_ops_or_scope.copy()
    try:
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()
  else:
    # Assumes that list_ops_or_scope is a list/tuple of ops with kwargs.
    if not isinstance(list_ops_or_scope, (list, tuple)):
      raise TypeError('list_ops_or_scope must either be a list/tuple or reused '
                      'scope (i.e. dict)')
    try:
      current_scope = current_arg_scope().copy()
      for op in list_ops_or_scope:
        key = arg_scope_func_key(op)
        if not has_arg_scope(op):
          raise ValueError('%s is not decorated with @add_arg_scope',
                           _name_op(op))
        if key in current_scope:
          current_kwargs = current_scope[key].copy()
          current_kwargs.update(kwargs)
          current_scope[key] = current_kwargs
        else:
          current_scope[key] = kwargs.copy()
      _get_arg_stack().append(current_scope)
      yield current_scope
    finally:
      _get_arg_stack().pop()


def add_arg_scope(func):
  """Decorates a function with args so it can be used within an arg_scope.

  Args:
    func: function to decorate.

  Returns:
    A tuple with the decorated function func_with_args().
  """

  def func_with_args(*args, **kwargs):
    current_scope = current_arg_scope()
    current_args = kwargs
    key_func = arg_scope_func_key(func)
    if key_func in current_scope:
      current_args = current_scope[key_func].copy()
      current_args.update(kwargs)
    return func(*args, **current_args)

  _add_op(func)
  setattr(func_with_args, '_key_op', arg_scope_func_key(func))
  return tf_decorator.make_decorator(func, func_with_args)


def has_arg_scope(func):
  """Checks whether a func has been decorated with @add_arg_scope or not.

  Args:
    func: function to check.

  Returns:
    a boolean.
  """
  return arg_scope_func_key(func) in _DECORATED_OPS


def arg_scoped_arguments(func):
  """Returns the list kwargs that arg_scope can set for a func.

  Args:
    func: function which has been decorated with @add_arg_scope.

  Returns:
    a list of kwargs names.
  """
  assert has_arg_scope(func)
  return _DECORATED_OPS[arg_scope_func_key(func)]
