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
"""Utilities for type-dependent behavior used in py2tf-generated code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.py2tf.utils.type_check import is_tensor
from tensorflow.python.ops import control_flow_ops


def dynamic_is(left, right):
  # TODO(alexbw) if we're sure we should leave 'is' in place,
  # then change the semantics in converters/logical_expressions.py
  return left is right


def dynamic_is_not(left, right):
  return left is not right


def run_cond(condition, true_fn, false_fn):
  """Type-dependent functional conditional.

  Args:
    condition: A Tensor or Python bool.
    true_fn: A Python callable implementing the true branch of the conditional.
    false_fn: A Python callable implementing the false branch of the
      conditional.

  Returns:
    result: The result of calling the appropriate branch. If condition is a
    Tensor, tf.cond will be used. Otherwise, a standard Python if statement will
    be ran.
  """
  if is_tensor(condition):
    return control_flow_ops.cond(condition, true_fn, false_fn)
  else:
    return py_cond(condition, true_fn, false_fn)


def py_cond(condition, true_fn, false_fn):
  if condition:
    return true_fn()
  else:
    return false_fn()


def run_while(cond_fn, body_fn, init_args):
  """Type-dependent functional while loop.

  Args:
    cond_fn: A Python callable implementing the stop conditions of the loop.
    body_fn: A Python callable implementing the body of the loop.
    init_args: The initial values of the arguments that will be passed to both
      cond_fn and body_fn.

  Returns:
    result: A list of values with the same shape and type as init_args. If any
    of the init_args, or any variables closed-over in cond_fn are Tensors,
    tf.while_loop will be used, otherwise a Python while loop will be ran.

  Raises:
    ValueError: if init_args is not a tuple or list with one or more elements.
  """
  if not isinstance(init_args, (tuple, list)) or not init_args:
    raise ValueError(
        'init_args must be a non-empty list or tuple, found %s' % init_args)

  # TODO(alexbw): statically determine all active variables in cond_fn,
  # and pass them directly
  closure_vars = tuple(
      [c.cell_contents for c in six.get_function_closure(cond_fn) or []])
  possibly_tensors = tuple(init_args) + closure_vars
  if is_tensor(*possibly_tensors):
    return control_flow_ops.while_loop(cond_fn, body_fn, init_args)
  else:
    return py_while_loop(cond_fn, body_fn, init_args)


def py_while_loop(cond_fn, body_fn, init_args):
  state = init_args
  while cond_fn(*state):
    state = body_fn(*state)
  return state
