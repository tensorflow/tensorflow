# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Conditional expressions (e.g. the ternary if statement)."""


import functools

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils


def if_exp(cond, if_true, if_false, expr_repr):
  if tensors.is_dense_tensor(cond):
    return _tf_if_exp(cond, if_true, if_false, expr_repr)
  else:
    return _py_if_exp(cond, if_true, if_false)


def _tf_if_exp(cond, if_true, if_false, expr_repr):
  """Overload of if_exp that stages a TF cond."""
  # TODO(mdan): Use nonlocal once we no longer need to support py2.
  true_val = []
  false_val = []

  def true_fn():
    true_val.append(if_true())
    if true_val and false_val:
      _verify_tf_if_exp_cond_var(expr_repr, true_val[0], false_val[0])
    return true_val[0]

  def false_fn():
    false_val.append(if_false())
    if true_val and false_val:
      _verify_tf_if_exp_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]

  return tf_cond.cond(cond, true_fn, false_fn)


def _verify_tf_if_exp_cond_var(expr_repr, true_val, false_val):
  """Verifies that both branches of a conditional expression are compatible."""
  try:
    nest.assert_same_structure(true_val, false_val, expand_composites=True)
  except (TypeError, ValueError):
    try:
      true_val = variable_utils.convert_variables_to_tensors(true_val)
      false_val = variable_utils.convert_variables_to_tensors(false_val)
      nest.assert_same_structure(true_val, false_val, expand_composites=True)
    except (TypeError, ValueError) as e:
      raise ValueError(
          'conditional expression "{}" must return the same nested structure '
          'in both branches. If you only need side effects, use a statement '
          '`if:` instead; otherwise, make both branches return the same '
          'structure or use `tf.cond` explicitly.'.format(expr_repr)) from e

  nest.map_structure(
      functools.partial(control_flow.verify_single_cond_var, expr_repr),
      true_val,
      false_val)


def _py_if_exp(cond, if_true, if_false):
  return if_true() if cond else if_false()
