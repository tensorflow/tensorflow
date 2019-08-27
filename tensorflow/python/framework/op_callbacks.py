# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Unified callbacks op execution and creation under eager and graph modes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.eager import context
from tensorflow.python.eager import execute

# A thread-local state object. It may hold the following attributes:
#   - `callbacks`: the thread-local stack of op callbacks.
#   - `invoking_callbacks`: a boolean used to keep track of whether
#     we are currently invoking an op_callback.
_state = threading.local()


class _OpCallbackContextManager(object):
  """Context manager for op callbacks."""

  def __init__(self, callback_fn):
    self._callback_fn = callback_fn

  def __enter__(self):
    """A method of when a scope of this context manager is being entered."""
    # Monkey-patch `execute.execute()`.
    execute.execute = execute.execute_with_callbacks
    if not hasattr(_state, "callback_stack"):
      _state.callback_stack = []
      _state.invoking_callbacks = False
    _state.callback_stack.append(self._callback_fn)

    ctx = context.context()
    if ctx.executing_eagerly():
      ctx.post_execution_callbacks.append(self._callback_fn)

  def __exit__(self, exec_type, exec_value, exec_traceback):
    """A method of when a scope of this context manager is being exited."""
    _state.callback_stack.pop()
    ctx = context.context()
    if ctx.executing_eagerly():
      ctx.post_execution_callbacks.pop()


def op_callback(callback_fn):
  r"""Intercepts op execution and op creation.

  The `callback_fn` will be invoked immediately after any of the three types
  of events:
    - The execution of an TensorFlow operation ("op" for short hereafter)
      under eager mode,
    - The execution of a FuncGraph under eager mode,
    - The creation of an op during graph construction (e.g., in
      @tf.function-decorated Python functions).

  Args:
    callback_fn: A callback_fn that has the following signature:
      def callback_fn(op_type,
                      inputs,
                      attrs,
                      outputs,
                      op_name=None,
                      graph=None):
        # op_type: The type of the op, as a string. E.g., "MatMul".
        #          For the special case of FuncGraph execution, op_type
        #          takes the name of the graph name, e.g.,
        #          "__inference_my_func_24".
        # inputs: (`tuple` of `Tensor`s) Input tensors to the op or the
        #         FuncGraph.
        #         - In eager execution, these are `EagerTensor`s.
        #         - In graph construction, these are non-eager `Tensor`s
        #           that form the inputs to the just-created op.
        # attrs: The attributes of the op or FuncGraph of which the execution
        #        or creation caused the current invocation of the callback.
        #        This is applicable to both eager- and graph-based execution,
        #        as well as graph construction.
        #        This is a tuple of alternating attribute keys and attribute
        #        values. E.g., `('adjoint_a', False, 'adjoint_b', False)`.
        # outputs: (`tuple of `Tensor`s) Output tensors from the op or
        #          FuncGraph.
        #          In eager execution, these are `EagerTensor`s.
        #          In graph construction, these are non-eager `Tensor`s that
        #          are the outputs of the just-created op.
        # op_name: Name of the op.
        #          - If the current invocation of the callback is due to the
        #            eager execution of an op or FuncGraph, this will be
        #            `None`, as op names are meaningless in eager execution.
        #          - In graph construction, this is the name of the op, e.g.,
        #            "MatMul_2".
        # graph: The graph that the op belongs to (if any).
        #        - In eager execution of an op or FuncGraph, this is `None`.
        #        - In graph construction, this is the op's containing graph
        #          as a `tf.Graph` object.
        #
        # Return values:
        #   This callback function is expected to return `None` or
        #   a `list` or `tuple` of `Tensor`s with its length matching
        #   `len(outputs)`, in the order that corresponds to that of the
        #   `outputs` argument.
        #   If the return value is `None`, downstream execution or graph
        #   construction will be unaffected.
        #   Howevevr, if the return value is a `list` or `tuple` of `Tensor`s,
        #   - In eager execution, these returned `Tensor`s should be
        #     `EagerTensor`s. Their values will replace the original values of
        #     `outputs` for downstream eager execution. (*Not implemented yet*).
        #   - In graph construction, these returned `Tensor`s should be
        #     non-eager `Tensor`s. Their values will replace the original
        #     `outputs` for downstream graph construction.

  Returns:
    A thread-local context manager. Within the scope of the context
    manager, all eager op/graph execution and graph op construction
    will invoke `callback_fn`.

  Raises:
    ValueEror: If `callback_fn` is not callable.
  """
  # TODO(b/139668041): Implement support for overriding `EagerTensor`s from
  # callback.
  if callback_fn is None:
    raise ValueError("Passed callback function cannot be None.")
  if not callable(callback_fn):
    raise ValueError(
        "Callback function passed to op_callback() is expected to be callable, "
        "but is not. Recevied %s" % callback_fn)
  return _OpCallbackContextManager(callback_fn)


def should_invoke_op_callbacks():
  """Determine if op callbacks are present and should be invoked.

  Returns:
    A thread-local result (boolean) indicating whether any op callback(s) exist
    and should be invoked.
  """
  return (
      hasattr(_state, "callback_stack") and _state.callback_stack and
      not (hasattr(_state, "invoking_callbacks") and _state.invoking_callbacks))


def invoke_op_callbacks(op_type,
                        inputs,
                        attrs,
                        outputs,
                        op_name=None,
                        graph=None):
  r"""Invoke the callbacks that exist in the current scope (if any).

  If no callbacks are present in the current scope, this method returns
  immediately.

  Args:
    op_type: Type of the operation (e.g., "MatMul").
    inputs: Input tensors to the op. These are `EagerTensor`s in the case of
      eager execution of ops or `FuncGraph`s, and are non-eager `Tensor`s in the
      case of graph construction.
    attrs: Attributes of the op, as `tuple` of alternating keys and values.
    outputs: Output tensors from the op. These are `EagerTensor`s in the case of
      eager execution and are non-eager `Tensor`s in the case of graph
      construction.
    op_name: Name of the op. Applicable if and only if this method is invoked
      due to the graph construction of an op or the eager execution of of a
      `FuncGraph`.
    graph: The graph involved (if any).
      - In the case if the eager execution of an op or FuncGraph, this is
        `None`.
      - In the case of the graph construction of an op, this is the `tf.Graph`
        object being built.

  Returns:
    `None`, or a `list` or `tuple` of output tenors that will override the
    original (input) `outputs`.
  """
  if _state.callback_stack:
    # Guards against stack overflow that can result from recursive invocation
    # due to op constructions inside client-supplied op callbacks.
    _state.invoking_callbacks = True
    try:
      if isinstance(attrs, dict):
        attrs_list = []
        for key in attrs:
          attrs_list.append(key)
          attrs_list.append(attrs[key])
        attrs_tuple = tuple(attrs_list)
      else:
        attrs_tuple = attrs

      new_outputs = outputs
      for callback in reversed(_state.callback_stack):
        new_outputs = callback(
            op_type,
            inputs,
            attrs_tuple,
            new_outputs,
            op_name=op_name,
            graph=graph)
        if new_outputs is not None and len(new_outputs) != len(outputs):
          raise ValueError(
              "The op callback returned %s tensors, which does not match the "
              "original number of outputs of op %s (%d)." %
              (len(new_outputs), op_name, len(outputs)))
      return new_outputs
    finally:
      _state.invoking_callbacks = False
  else:
    return outputs
