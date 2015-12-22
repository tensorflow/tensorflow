# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Provides templates which allow variable sharing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import traceback

from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import logging


__all__ = ["make_template"]


def make_template(name_, func_, **kwargs):
  """Given an arbitrary function, wrap it so that it does variable sharing.

  This wraps `func_` in a Template and partially evaluates it. Templates are
  functions that create variables the first time they are called and reuse them
  thereafter. In order for `func_` to be compatible with a `Template` it must
  have the following properties:

  * The function should create all trainable variables and any variables that
     should be reused by calling `tf.get_variable`. If a trainable variable is
     created using `tf.Variable`, then a ValueError will be thrown. Variables
     that are intended to be locals can be created by specifying
     `tf.Variable(..., trainable=false)`.
  * The function may use variable scopes and other templates internally to
      create and reuse variables, but it shouldn't use `tf.get_variables` to
      capture variables that are defined outside of the scope of the function.
  * Internal scopes and variable names should not depend on any arguments that
      are not supplied to `make_template`. In general you will get a ValueError
      telling you that you are trying to reuse a variable that doesn't exist
      if you make a mistake.

  In the following example, both `z` and `w` will be scaled by the same `y`. It
  is important to note that if we didn't assign `scalar_name` and used a
  different name for z and w that a `ValueError` would be thrown because it
  couldn't reuse the variable.

  ```python
  def my_op(x, scalar_name):
    var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
    return x * var1

  scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')

  z = scale_by_y(input1)
  w = scale_by_y(input2)
  ```

  As a safe-guard, the returned function will raise a `ValueError` after the
  first call if trainable variables are created by calling `tf.Variable`.

  If all of these are true, then 2 properties are enforced by the template:

  1. Calling the same template multiple times will share all non-local
      variables.
  2. Two different templates are guaranteed to be unique, unless you reenter the
      same variable scope as the initial definition of a template and redefine
      it. An examples of this exception:

  ```python
  def my_op(x, scalar_name):
    var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
    return x * var1

  with tf.variable_scope('scope') as vs:
    scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')
    z = scale_by_y(input1)
    w = scale_by_y(input2)

  # Creates a template that reuses the variables above.
  with tf.variable_scope(vs, reuse=True):
    scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
    z2 = scale_by_y2(input1)
    w2 = scale_by_y2(input2)
  ```

  Note: The full variable scope is captured at the time of the first call.

  Note: `name_` and `func_` have a following underscore to reduce the likelihood
  of collisions with kwargs.

  Args:
    name_: A name for the scope created by this template. If necessary, the name
      will be made unique by appending `_N` to the name.
    func_: The function to wrap.
    **kwargs: Keyword arguments to apply to `func_`.

  Returns:
    A function that will enter a `variable_scope` before calling `func_`. The
    first time it is called, it will create a non-reusing scope so that the
    variables will be unique.  On each subsequent call, it will reuse those
    variables.

  Raises:
    ValueError: if the name is None.
  """
  if kwargs:
    func_ = functools.partial(func_, **kwargs)
  return Template(name_, func_)


def _skip_common_stack_elements(stacktrace, base_case):
  """Skips items that the target stacktrace shares with the base stacktrace."""
  for i, (trace, base) in enumerate(zip(stacktrace, base_case)):
    if trace != base:
      return stacktrace[i:]
  return stacktrace[-1:]


class Template(object):
  """Wrap a function to aid in variable sharing.

  Templates are functions that create variables the first time they are called
  and reuse them thereafter. See `make_template` for full documentation.

  Note: The full variable scope is captured at the time of the first call.
  """

  def __init__(self, name, func):
    """Creates a template for the given function.

    Args:
      name: A name for the scope created by this template. The
        name will be made unique by appending `_N` to the it (see how
        `tf.variable_op_scope` treats the `default_name` for details).
      func: The function to apply each time.

    Raises:
      ValueError: if the name is None.
    """
    self._func = func
    self._stacktrace = traceback.format_stack()[:-2]
    self._name = name
    if name is None:
      raise ValueError("name cannot be None.")
    self._var_scope = None

  def _call_func(self, args, kwargs, check_for_new_variables):
    try:
      vars_at_start = len(ops.get_collection(ops.GraphKeys.VARIABLES))
      trainable_at_start = len(
          ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

      result = self._func(*args, **kwargs)
      if check_for_new_variables:
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        # If a variable that we intend to train is created as a side effect
        # of creating a template, then that is almost certainly an error.
        if trainable_at_start != len(trainable_variables):
          raise ValueError("Trainable variable created when calling a template "
                           "after the first time, perhaps you used tf.Variable "
                           "when you meant tf.get_variable: %s" %
                           (trainable_variables[trainable_at_start:],))

        # Non-trainable tracking variables are a legitimate reason why a new
        # variable would be created, but it is a relatively advanced use-case,
        # so log it.
        variables = ops.get_collection(ops.GraphKeys.VARIABLES)
        if vars_at_start != len(variables):
          logging.info("New variables created when calling a template after "
                       "the first time, perhaps you used tf.Variable when you "
                       "meant tf.get_variable: %s",
                       variables[vars_at_start:])
      return result
    except Exception as exc:
      # Reraise the exception, but append the original definition to the
      # trace.
      args = exc.args
      if not args:
        arg0 = ""
      else:
        arg0 = args[0]
      trace = "".join(_skip_common_stack_elements(self._stacktrace,
                                                  traceback.format_stack()))
      arg0 = "%s\n\noriginally defined at:\n%s" % (arg0, trace)
      new_args = [arg0]
      new_args.extend(args[1:])
      exc.args = tuple(new_args)
      raise

  def __call__(self, *args, **kwargs):
    # Capture the name of the variable_scope here because if we capture at
    # construction, then name_scopes would have a '_N+1' suffix.
    if self._var_scope:
      with variable_scope.variable_scope(self._var_scope, reuse=True):
        return self._call_func(args, kwargs, check_for_new_variables=True)
    else:
      with variable_scope.variable_op_scope([], None, self._name) as vs:
        self._var_scope = vs
        return self._call_func(args, kwargs, check_for_new_variables=False)
