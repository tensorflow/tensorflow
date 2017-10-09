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

from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator


def custom_gradient(f):
  """Decorator to define a function with a custom gradient.

  The input function is expected to return the tuple
    (results, gradient_function).

  The output function will return results while possibly recording the
  gradient_function and inputs in the tape.

  Args:
    f: function to be decorated.

  Returns:
    decorated function.
  """

  def decorated(*args, **kwargs):
    """Decorated function with custom gradient."""
    if context.in_graph_mode():
      if kwargs:
        raise ValueError(
            "custom_gradient in graph mode doesn't support keyword arguments.")
      name = "CustomGradient-%s" % tf_ops.uid()
      args = [tf_ops.convert_to_tensor(x) for x in args]
      result, grad_fn = f(*args)
      flat_result = nest.flatten(result)
      all_tensors = flat_result + args

      @tf_ops.RegisterGradient(name)
      def internal_grad_fn(unused_op, *result_grads):  # pylint: disable=unused-variable
        gradients = nest.flatten(grad_fn(*result_grads[:len(flat_result)]))
        # Need to return one value per input to the IdentityN, so pad the
        # gradients of the inputs of the custom_gradient function with the
        # gradients of the outputs as well.
        return ([None] * len(flat_result)) + gradients

      with tf_ops.get_default_graph().gradient_override_map(
          {"IdentityN": name}):
        all_tensors = array_ops.identity_n(all_tensors)
      return nest.pack_sequence_as(
          structure=result, flat_sequence=all_tensors[:len(flat_result)])

    input_tensors = [x for x in args
                     if isinstance(x, tf_ops.Tensor)]

    with tape.stop_recording():
      result, grad_fn = f(*args, **kwargs)

    # TODO(apassos): naive uses of custom_gradient will not get the correct
    # second derivative this way if they capture any output tensors. Change the
    # signature of custom_gradient.
    def actual_grad_fn(*outputs):
      return nest.flatten(grad_fn(*outputs))

    flat_result = nest.flatten(result)
    tape.record_operation(
        f.__name__,
        flat_result,
        input_tensors,
        [],
        actual_grad_fn)
    flat_result = list(flat_result)
    return result

  return tf_decorator.make_decorator(f, decorated)
