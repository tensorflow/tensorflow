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
"""Utilities for type-dependent behavior used in autograph-generated code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.utils.type_check import is_tensor
from tensorflow.python.ops import control_flow_ops


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
  """Functional version of Python's conditional."""
  if condition:
    results = true_fn()
  else:
    results = false_fn()

  # The contract for the branch functions is to return tuples, but they should
  # be collapsed to a single element when there is only one output.
  if len(results) == 1:
    return results[0]
  return results
