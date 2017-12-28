# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated


__all__ = ["make_template"]


def make_template(name_, func_, create_scope_now_=False, unique_name_=None,
                  custom_getter_=None, **kwargs):
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
      create and reuse variables, but it shouldn't use `tf.global_variables` to
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

  Depending on the value of `create_scope_now_`, the full variable scope may be
  captured either at the time of first call or at the time of construction. If
  this option is set to True, then all Tensors created by repeated calls to the
  template will have an extra trailing _N+1 to their name, as the first time the
  scope is entered in the Template constructor no Tensors are created.

  Note: `name_`, `func_` and `create_scope_now_` have a trailing underscore to
  reduce the likelihood of collisions with kwargs.

  Args:
    name_: A name for the scope created by this template. If necessary, the name
      will be made unique by appending `_N` to the name.
    func_: The function to wrap.
    create_scope_now_: Boolean controlling whether the scope should be created
      when the template is constructed or when the template is called. Default
      is False, meaning the scope is created when the template is called.
    unique_name_: When used, it overrides name_ and is not made unique. If a
      template of the same scope/unique_name already exists and reuse is false,
      an error is raised. Defaults to None.
    custom_getter_: Optional custom getter for variables used in `func_`. See
      the @{tf.get_variable} `custom_getter` documentation for
      more information.
    **kwargs: Keyword arguments to apply to `func_`.

  Returns:
    A function to encapsulate a set of variables which should be created once
    and reused. An enclosing scope will created, either where `make_template`
    is called, or wherever the result is called, depending on the value of
    `create_scope_now_`. Regardless of the value, the first time the template
    is called it will enter the scope with no reuse, and call `func_` to create
    variables, which are guaranteed to be unique. All subsequent calls will
    re-enter the scope and reuse those variables.

  Raises:
    ValueError: if the name is None.
  """
  if kwargs:
    func_ = functools.partial(func_, **kwargs)
  if context.in_eager_mode():
    return EagerTemplate(
        name_, func_, create_scope_now=create_scope_now_,
        unique_name=unique_name_, custom_getter=custom_getter_)
  return Template(
      name_, func_, create_scope_now=create_scope_now_,
      unique_name=unique_name_, custom_getter=custom_getter_)


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

  Note: By default, the full variable scope is captured at the time of first
  call. If `create_scope_now_` is passed as True to the constructor, the full
  scope will be captured there, but no variables will created until the first
  call.
  """

  def __init__(self, name, func, create_scope_now=False, unique_name=None,
               custom_getter=None):
    """Creates a template for the given function.

    Args:
      name: A name for the scope created by this template. The
        name will be made unique by appending `_N` to the it (see how
        `tf.variable_scope` treats the `default_name` for details).
      func: The function to apply each time.
      create_scope_now: Whether to create the scope at Template construction
        time, rather than first call. Defaults to false. Creating the scope at
        construction time may be more convenient if the template is to passed
        through much lower level code, and you want to be sure of the scope
        name without knowing exactly where it will be first called. If set to
        True, the scope will be created in the constructor, and all subsequent
        times in __call__, leading to a trailing numeral being added to the
        names of all created Tensors. If set to False, the scope will be created
        at the first call location.
      unique_name: When used, it overrides name_ and is not made unique. If a
        template of the same scope/unique_name already exists and reuse is
        false, an error is raised. Defaults to None.
      custom_getter: optional custom getter to pass to variable_scope()

    Raises:
      ValueError: if the name is None.
    """
    self._func = func
    self._stacktrace = traceback.format_stack()[:-2]
    self._name = name
    self._unique_name = unique_name
    self._custom_getter = custom_getter
    if name is None:
      raise ValueError("name cannot be None.")
    if create_scope_now:
      with variable_scope._pure_variable_scope(  # pylint:disable=protected-access
          (self._unique_name or
           variable_scope._get_unique_variable_scope(self._name)),  # pylint:disable=protected-access
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
    else:
      self._variable_scope = None
    # This variable keeps track of whether the template has been called yet,
    # which is not the same as whether the scope has been created.
    self._variables_created = False

  def _call_func(self, args, kwargs, check_for_new_variables):
    try:
      vars_at_start = len(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
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
        variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
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
    if self._variable_scope:
      if self._variables_created:
        # This is not the first visit to __call__, so variables have already
        # been created, and we want to reuse them.
        with variable_scope.variable_scope(self._variable_scope, reuse=True):
          return self._call_func(args, kwargs, check_for_new_variables=True)
      else:
        # This is the first visit to __call__, but the scope has already been
        # created in the constructor. Set _variables_created after the inner
        # function is successfully called so that subsequent calls take the if
        # branch above.
        with variable_scope.variable_scope(self._variable_scope):
          result = self._call_func(args, kwargs, check_for_new_variables=False)
          self._variables_created = True
          return result
    else:
      # The scope was not created at construction time, so create it here.
      # Subsequent calls should reuse variables.
      with variable_scope.variable_scope(
          self._unique_name, self._name,
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
        result = self._call_func(args, kwargs, check_for_new_variables=False)
        self._variables_created = True
        return result

  @property
  def name(self):
    """Returns the name given to this Template."""
    return self._name

  @property
  def func(self):
    """Returns the func given to this Template."""
    return self._func

  @property
  def variable_scope(self):
    """Returns the variable scope object created by this Template."""
    return self._variable_scope

  @property
  def variable_scope_name(self):
    """Returns the variable scope name created by this Template."""
    if self._variable_scope:
      name = self._variable_scope.name
      # To prevent partial matches on the scope_name, we add '/' at the end.
      return name if name[-1] == "/" else name + "/"

  @property
  def variables(self):
    """Returns the list of global and local variables created by the Template.
    """
    return self.global_variables + self.local_variables

  @property
  def trainable_variables(self):
    """Returns the list of trainable variables created by the Template."""
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                self.variable_scope_name)
    else:
      return []

  @property
  def non_trainable_variables(self):
    """Returns the list of non-trainable variables created by the Template."""
    # TODO(apassos) Make sure it matches Eager when using local variables.
    global_variables = self.global_variables
    trainable_variables = set(self.trainable_variables)
    return [x for x in global_variables if x not in trainable_variables]

  @property
  def global_variables(self):
    """Returns the list of global variables created by the Template."""
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                self.variable_scope_name)
    else:
      return []

  @property
  def local_variables(self):
    """Returns the list of global variables created by the Template."""
    if self._variables_created:
      return ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES,
                                self.variable_scope_name)
    else:
      return []

  @property
  def weights(self):
    """List of weights/variables created by the Template."""
    return self.variables

  @property
  def trainable_weights(self):
    """List of trainable weights/variables created by the Template."""
    return self.trainable_variables

  @property
  def non_trainable_weights(self):
    """List of non-trainable weights/variables created by the Template."""
    return self.non_trainable_variables

  @property
  @deprecated(
      "2017-02-21", "The .var_scope property is deprecated. Please change your "
      "code to use the .variable_scope property")
  def var_scope(self):
    """Returns the variable scope object created by this Template."""
    return self._variable_scope


class EagerTemplate(Template):
  """Wrap a function to aid in variable sharing in Eager mode.

  Templates are functions that create variables the first time they are called
  and reuse them thereafter. See `make_template` for full documentation.

  Note: By default, the full variable scope is captured at the time of first
  call. If `create_scope_now` is passed as True to the constructor, the full
  scope will be captured there, but no variables will be created until the first
  call.
  """

  def __init__(self, name, func, create_scope_now=False, unique_name=None,
               custom_getter=None):
    """Creates a template for the given function.

    Args:
      name: A name for the scope created by this template. The
        name will be made unique by appending `_N` to the it (see how
        `tf.variable_scope` treats the `default_name` for details).
      func: The function to apply each time.
      create_scope_now: Whether to create the scope at Template construction
        time, rather than first call. Defaults to false. Creating the scope at
        construction time may be more convenient if the template is passed
        through much lower level code, and you want to be sure of the scope
        name without knowing exactly where it will be first called. If set to
        True, the scope will be created in the constructor, and all subsequent
        times in __call__, leading to a trailing numeral being added to the
        names of all created Tensors. If set to False, the scope will be created
        at the first call location.
      unique_name: When used, it overrides name_ and is not made unique. If a
        template of the same scope/unique_name already exists and reuse is
        false, an error is raised. Defaults to None.
      custom_getter: optional custom getter to pass to variable_scope()

    Raises:
      RuntimeError: if eager mode is not enabled.
      ValueError: if the name is None or unique_name is provided.
    """
    if not context.in_eager_mode():
      raise RuntimeError(
          "{} objects can only be used when eager execution is enabled, use "
          "tf.Template for graph construction".
          format(type(self)))
    if unique_name:
      raise ValueError("unique_name cannot be used in eager mode.")
    super(EagerTemplate, self).__init__(name, func, create_scope_now,
                                        unique_name, custom_getter)
    # Create an eager variable store only if the current variable store cannot
    # store eager variables. This should allow for correct nesting.
    default_vstore = variable_scope._get_default_variable_store()  # pylint: disable=protected-access
    if default_vstore._store_eager_variables:  # pylint: disable=protected-access
      raise ValueError("Nested EagerTemaplates are not currently supported.")
    else:
      self._eager_variable_store = variable_scope.EagerVariableStore()

  def _call_func(self, args, kwargs, check_for_new_variables):
    try:
      vars_at_start = self._eager_variable_store.variables()
      trainable_at_start = self._eager_variable_store.trainable_variables()

      result = self._func(*args, **kwargs)
      if check_for_new_variables:
        trainable_variables = self._eager_variable_store.trainable_variables()
        # If a variable that we intend to train is created as a side effect
        # of creating a template, then that is almost certainly an error.
        if len(trainable_at_start) != len(trainable_variables):
          raise ValueError("Trainable variable created when calling a template "
                           "after the first time, perhaps you used tf.Variable "
                           "when you meant tf.get_variable: %s" %
                           list(set(trainable_variables) -
                                set(trainable_at_start)))

        # Non-trainable tracking variables are a legitimate reason why a new
        # variable would be created, but it is a relatively advanced use-case,
        # so log it.
        variables = self._eager_variable_store.variables()
        if len(vars_at_start) != len(variables):
          logging.info("New variables created when calling a template after "
                       "the first time, perhaps you used tf.Variable when you "
                       "meant tf.get_variable: %s",
                       list(set(variables) - set(vars_at_start)))
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
    if self._variable_scope:
      if self._variables_created:
        # This is not the first visit to __call__, so variables have already
        # been created, and we want to reuse them.
        with variable_scope.variable_scope(self._variable_scope,
                                           reuse=variable_scope.AUTO_REUSE):
          with self._eager_variable_store.as_default():
            return self._call_func(args, kwargs, check_for_new_variables=True)
      else:
        # This is the first visit to __call__, but the scope has already been
        # created in the constructor. Set _variables_created after the inner
        # function is successfully called so that subsequent calls take the if
        # branch above.
        with variable_scope.variable_scope(self._variable_scope,
                                           reuse=variable_scope.AUTO_REUSE):
          with self._eager_variable_store.as_default():
            result = self._call_func(args, kwargs,
                                     check_for_new_variables=False)
        self._variables_created = True
        return result
    else:
      # The scope was not created at construction time, so create it here.
      # Subsequent calls should reuse variables.
      with variable_scope.variable_scope(
          self._unique_name, self._name,
          custom_getter=self._custom_getter) as vs:
        self._variable_scope = vs
        with self._eager_variable_store.as_default():
          result = self._call_func(args, kwargs,
                                   check_for_new_variables=False)
        self._variables_created = True
        return result

  @property
  def name(self):
    """Returns the name given to this Template."""
    return self._name

  @property
  def func(self):
    """Returns the func given to this Template."""
    return self._func

  @property
  def variable_scope(self):
    """Returns the variable scope object created by this Template."""
    return self._variable_scope

  @property
  def variable_scope_name(self):
    """Returns the variable scope name created by this Template."""
    if self._variable_scope:
      name = self._variable_scope.name
      # To prevent partial matches on the scope_name, we add '/' at the end.
      return name if name[-1] == "/" else name + "/"

  @property
  def variables(self):
    """Returns the list of variables created by the Template."""
    # Currently there is no local variable in Eager mode.
    return self._eager_variable_store.variables()

  @property
  def trainable_variables(self):
    """Returns the list of trainable variables created by the Template."""
    # Currently there is no local variable in Eager mode.
    return self._eager_variable_store.trainable_variables()

  @property
  def non_trainable_variables(self):
    """Returns the list of non-trainable variables created by the Template."""
    # Currently there is no local variable in Eager mode.
    return self._eager_variable_store.non_trainable_variables()

  @property
  def global_variables(self):
    """Returns the list of global variables created by the Template."""
    # Currently there is no local variable in Eager mode.
    return self.variables

  @property
  def local_variables(self):
    """Returns the list of global variables created by the Template."""
    # Currently there is no local variable in Eager mode.
    return []
