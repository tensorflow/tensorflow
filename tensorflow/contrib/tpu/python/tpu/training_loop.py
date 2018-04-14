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
# =============================================================================

"""Library for constructing a training loop, suitable for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import tpu_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


def while_loop(condition, body, inputs=None, infeed_queue=None, name=None):
  """Builds a training loop for TPUs.

  The set of loop-carried tensors corresponds to `inputs`.  Both
  `condition` and `body` take the current value of the loop-carried
  tensors. 'body' additionally takes a tuple of infeed from
  infeed_queue if infeed_queue is not None. `condition` must return a
  single boolean value that determines whether iteration
  continues. `body` must return an updated list of values for the
  loop-carried tensors.

  Args:
    condition: a Python function that builds the loop condition.
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the training loop, or
      None (equivalent to an empty list).
    infeed_queue: if not None, the infeed queue from which to append a tuple
      of arguments as inputs to condition.
    name: (Deprecated) Does nothing.

  Returns:
    The final values of the loop-carried tensors.

  Raises:
    TypeError: if body or condition has the wrong signature.
  """
  del name
  # Converts inputs to Tensors.
  inputs = [] if inputs is None else [ops.convert_to_tensor(x) for
                                      x in inputs]
  input_types = [x.dtype for x in inputs]
  input_arity = len(inputs)

  body_arg_error = tpu_function.check_function_argument_count(
      body, input_arity, infeed_queue)
  if body_arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied loop body function cannot be called with the specified "
          "inputs. You specified %d inputs: %s, but the loop body needs %s" % (
              input_arity, str([i.name for i in inputs]), body_arg_error))
    else:
      raise TypeError(
          "Supplied loop body function cannot be called with the specified "
          "inputs. You specified %d inputs: %s and %d additional inputs from "
          "infeed, but the computation needs %s" % (input_arity, str(
              [i.name for i in inputs]), infeed_queue.number_of_tuple_elements,
                                                    body_arg_error))
  condition_arg_error = tpu_function.check_function_argument_count(
      condition, input_arity, None)
  if condition_arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied loop condition function cannot be called with the "
          "specified inputs. You specified %d inputs: %s, but the loop "
          "condition needs %s" % (input_arity, str([i.name for i in inputs]),
                                  condition_arg_error))
    else:
      raise TypeError(
          "Supplied loop condition function cannot be called with the "
          "specified inputs. You specified %d inputs: %s, but the loop "
          "condition needs %s. Note that infeed is not passed to the loop "
          "condition." % (input_arity, str([i.name for i in inputs]),
                          condition_arg_error))

  def condition_wrapper(*inputs):
    # Discards the dummy output added for arity-0 loops.
    if input_arity == 0:
      inputs = []
    return condition(*inputs)

  def body_wrapper(*inputs):
    """Wrapper around `body` that handles infeed queues and control deps."""
    inputs = list(inputs)

    # Discards the dummy output added for arity-0 loops.
    if input_arity == 0:
      inputs = []

    # Runs `body` with the dequeue_ops appended.
    if infeed_queue:
      number_of_shards = tpu_function.get_tpu_context().number_of_shards
      if number_of_shards is None:
        raise ValueError("Can't build training loop with infeed when there is "
                         "no tpu_shard_context. Are you building a loop or "
                         "graph directly rather than from inside tpu.rewrite, "
                         "tpu.batch_parallel, tpu.shard, or tpu.replicate?")
      infeed_queue.set_number_of_shards(number_of_shards)
      dequeue_ops = [d for d in infeed_queue.generate_dequeue_op()]
    else:
      dequeue_ops = []
    outputs = body(*(inputs + dequeue_ops))

    # If the computation only returned one value, make it a tuple.
    if not isinstance(outputs, (list, tuple)):
      outputs = (outputs,)

    outputs = [
        o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o)
        for o in outputs
    ]

    # Separates the returned Operations and Tensors.
    output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
    output_tensors = [o for o in outputs
                      if not isinstance(o, ops.Operation)]

    if outputs != output_tensors + output_operations:
      raise ValueError(
          "TPU training loop body must return zero or more Tensor values "
          "followed by zero or more Operations.")

    output_types = [op.dtype for op in output_tensors]
    if input_types != output_types:
      raise TypeError(
          "Mismatch between input types and output types for training loop "
          "body: {} vs {}".format(input_types, output_types))

    # Add the dequeue operations to output_operations to ensure they are run
    # by the loop, even if the programmer's loop body does not use them.
    output_operations += dequeue_ops

    # Add a dummy output, if needed.
    if not output_tensors:
      output_tensors = array_ops.constant(0)

    if output_operations:
      # TODO(phawkins): in principle this is too restrictive since it serializes
      # the training loop steps. In practice it does not matter since this loop
      # will be compiled by XLA.
      return control_flow_ops.tuple(output_tensors,
                                    control_inputs=output_operations)
    else:
      return output_tensors

  # If the body has arity 0, add a dummy loop-carried value to which we can add
  # control dependencies from any side-effecting operations.
  if input_arity == 0:
    inputs = [array_ops.constant(0)]
  return control_flow_ops.while_loop(condition_wrapper, body_wrapper, inputs,
                                     name="")


def repeat(n, body, inputs=None, infeed_queue=None, name=None):
  """Builds a training loop that executes a fixed number of iterations.

  The set of loop-carried tensors correspond to `inputs`.
  `body` must be a function that takes and returns the values of the
  loop-carried tensors.

  Args:
    n: the number of loop iterations
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the training loop or
      None (equivalent to an empty list).
    infeed_queue: if not None, the infeed queue from which to append a tuple
      of arguments as inputs to condition.
    name: (Deprecated) Does nothing.
  Returns:
    The final values of the loop-carried tensors.
  Raises:
    ValueError: if there is a type error.
  """
  def _convert_to_list(xs):
    if not isinstance(xs, (list, tuple)):
      return [xs]
    else:
      return list(xs)

  def cond(i, *args):
    del args
    return i < n

  def body_wrapper(i, *args):
    return [i + 1] + _convert_to_list(body(*args))

  inputs = [0] if inputs is None else [0] + _convert_to_list(inputs)
  outputs = while_loop(
      cond, body_wrapper, inputs=inputs, infeed_queue=infeed_queue, name=name)
  outputs = _convert_to_list(outputs)
  if len(outputs) == 1:
    # Returns the Op rather than an empty list.
    return outputs[0].op
  else:
    return outputs[1:]
