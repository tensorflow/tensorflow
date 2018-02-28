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
"""Utility functions related to managing `tf.Variable`s.

@@externalize_variables_as_args
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl as gradients_ops
from tensorflow.python.ops import variable_scope as varscope_ops
from tensorflow.python.ops import variables as variables_ops

__all__ = [
    "externalize_variables_as_args",
]


# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.simplefilter("always")


def externalize_variables_as_args(fn,
                                  fn_args=(),
                                  ancestor_variables=None,
                                  possible_ancestor_vars=None,
                                  assert_variable_override=False,
                                  name=None):
  """"Converts variables within a callable into explicit args.

  Makes a new callable from `fn` which has arguments `list(fn_args) +
  list(ancestor_variables)`. If `ancestor_variables` is not specified, it is
  inferred by checking which of `possible_ancestor_vars` actually influences the
  return value of `fn` (concretely, gradient of `fn(*fn_args)` is not `None`).
  By default `possible_ancestor_vars` is `tf.trainable_variables() +
  tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)`.

  #### Examples:

  ```python
  num_samples = 2
  num_dims = 1
  dtype = np.float32

  def foo(x):
    x = tf.convert_to_tensor(x, dtype=dtype, name="x")
    s = x.shape.as_list()
    y = tf.get_variable(
        name="y",
        dtype=dtype,
        initializer=np.arange(np.prod(s)).reshape(s).astype(dtype))
    return x + y

  x = tf.constant(dtype([0.1, 0.2]))

  wrapped_foo, discovered_ancestor_variables = (
      externalize_variables_as_args(foo, [x]))

  new_x = dtype([[1.], [2.]])
  new_y = dtype([[3.], [4.]])
  new_result = wrapped_foo(new_x, new_y)
  # ==> [[4.], [6.]]

  discovered_ancestor_variables == [tf.get_variable("y", dtype)]
  # ==> [True]
  ```

  Args:
    fn: Python callable which returns a `Tensor` and accepts `*fn_args`.
    fn_args: Python list of args to `fn`. Represents dummy arguments passed to
      `fn` to trace its execution; actual values are unimportant. These args are
      only used to construct the output of `fn` and to resolve the ancestor
      `tf.Variable`s.
      Default value: `()` (i.e., `fn` takes no args).
    ancestor_variables: Python list of `tf.Variable`s. When `None` the list is
      expanded to non-`None` gradients of `fn(*fn_args)`. By directly providing
      the `ancestor_variables` the internal call to `fn` is avoided.
      Default value: `None` (i.e., `tf.Variable` dependencies are discovered).
    possible_ancestor_vars: Python list of possible `tf.Variable`s which might
      be a dependency of computing `fn(*fn_args)`.
      Default value: `None` (i.e., expanded as described above).
    assert_variable_override: Python `bool` indicating that not finding a
      `tf.Variable` in the override list is an exception.
      Default value: `False` (i.e., missing a `Variable` triggers a `warning`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "externalize_variables_as_args").

  Returns:
    wrapped_fn: Python callable taking arguments like
      `*(list(fn_args) + discovered_ancestor_variables)`.
    discovered_ancestor_variables: Python list of `tf.Variable`s known to be a
      dependency of `fn(*fn_args)`.

  Raises:
    ValueError: if `assert_variable_override` is `True` and `Variable` is
      requested but not overridden.
  """
  def _make_bypassing_custom_getter_fn(new_var_dict):
    """Return dict value rather than what would otherwise be dict key."""
    def _custom_getter(getter, *args, **kwargs):
      v = getter(*args, **kwargs)
      new_v = new_var_dict.get(v, None)
      if new_v is None:
        msg = "Variable \"{}\" not found in bypass dict.".format(v)
        if assert_variable_override:
          raise ValueError(msg)
        warnings.warn(msg)
        return v
      return new_v
    return _custom_getter

  with ops.name_scope(name, "externalize_variables_as_args"):
    if ancestor_variables is not None and not ancestor_variables:
      return fn, ()
    if ancestor_variables is None:
      y = fn(*fn_args)  # Side-effect: adds trainable vars.
      if possible_ancestor_vars is None:
        possible_ancestor_vars = (
            variables_ops.trainable_variables() +
            ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
      # TODO(b/72873296): Add a dedicated op for identifying ancestors.
      ancestors = [v for g, v
                   in zip(gradients_ops.gradients(y, possible_ancestor_vars),
                          possible_ancestor_vars)
                   if g is not None]
      ancestor_variables = sorted(ancestors, key=lambda v: v.name)
  n = len(fn_args)
  def _fn(*args):
    with ops.name_scope("wrapped_fn"):
      vars_dict = dict(
          (k, ops.convert_to_tensor(
              v, dtype=k.dtype.base_dtype, name=k.op.name))
          for k, v in zip(ancestor_variables, args[n:]))
      with varscope_ops.variable_scope(
          varscope_ops.get_variable_scope(),
          reuse=True,
          custom_getter=_make_bypassing_custom_getter_fn(vars_dict)):
        return fn(*args[:n])
  return _fn, ancestor_variables
