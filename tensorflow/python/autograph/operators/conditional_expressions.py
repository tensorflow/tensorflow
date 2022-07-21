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


from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import control_flow_ops


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
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return true_val[0]

  def false_fn():
    false_val.append(if_false())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]

  return control_flow_ops.cond(cond, true_fn, false_fn)


def _py_if_exp(cond, if_true, if_false):
  return if_true() if cond else if_false()
