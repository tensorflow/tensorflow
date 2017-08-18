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
"""Decorator to overrides the gradient for a function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autograd import core as ag_core

from tensorflow.python.eager import tape
from tensorflow.python.eager import tensor as _tensor
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.util import nest


def _watch_value_from_tape(tensor):
  for t in tape._tape_stack.stack:  # pylint: disable=protected-access
    w = t.value.tensors.get(tf_ops.tensor_id(tensor), None)
    if w is not None:
      return w
  return tensor


def custom_gradient(f):
  """Decorator to define a function with a custom gradient.

  The input function is expected to return the tuple
    (results, gradient_function)

  The output function will return results while possibly recording the
  gradient_function and inputs in the tape.

  Args:
    f: function to be decorated.

  Returns:
    decorated function.
  """

  def decorated(*args, **kwargs):
    """Decorated function with custom gradient."""
    input_tensors = [_watch_value_from_tape(x) for x in args
                     if isinstance(x, (_tensor.Tensor, tf_ops.Tensor))
                     or ag_core.isnode(x)]
    result, grad_fn = f(*args, **kwargs)

    flat_result = nest.flatten(result)
    flat_result = [ag_core.getval(x) for x in flat_result]
    flat_result = tape.record_operation(
        flat_result,
        input_tensors,
        [],
        grad_fn)
    flat_result = list(flat_result)
    return nest.pack_sequence_as(structure=result, flat_sequence=flat_result)

  return decorated
