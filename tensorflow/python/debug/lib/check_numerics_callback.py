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
"""Eager-graph unified check numerics callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np

from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import op_callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.util import compat


def limit_string_length(string, max_len=50):
  """Limit the length of input string.

  Args:
    string: Input string.
    max_len: (int or None) If int, the length limit. If None, no limit.

  Returns:
    Possibly length-limited string.
  """
  if max_len is None or len(string) <= max_len:
    return string
  else:
    return "..." + string[len(string) - max_len:]


_CHECK_NUMERICS_CALLBACK_SKIP_OPS = (
    # TODO(b/139668453): The following skipped ops are related to a limitation
    # in the op callback.
    b"Identity",
    b"If",
    b"StatelessIf",
    b"While",
)

# A dictionary that supports looking up the original input tensor names.
_CHECK_NUMERICS_INPUT_LOOKUP = collections.defaultdict(dict)


def _maybe_lookup_original_input_tensor(graph, tensor):
  if (graph and
      graph.name and
      graph.name in _CHECK_NUMERICS_INPUT_LOOKUP and
      tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph.name]):
    return _CHECK_NUMERICS_INPUT_LOOKUP[graph.name][tensor.name]
  else:
    return tensor


def get_check_numerics_error_message(slot,
                                     num_outputs,
                                     op_type,
                                     tensor,
                                     inputs,
                                     graph=None,
                                     traceback=None,
                                     stack_height_limit=30,
                                     path_length_limit=50):
  """Create a meaningful and user-friendly error message about offending tensor.

  The error message reveals the following info about the op that outputs
  NaN/Infinity: dtype, shape (to the extent known at graph-construction time),
  input tensors, stack trace for op creation (if is graph mode).

  Args:
    slot: (int) slot index of the tensor output.
    num_outputs: (int) total number of outputs of the op.
    op_type: (str) Type of the that generates `tensor`.
    tensor: (Tensor) the offending tensor, i.e., the tensor that contains
      Infinities or NaNs.
    inputs: (array of Tensor) inputs to the op that generates `tensor`.
    graph: (tf.Graph) the graph object that `tensor` belongs to. Available only
      under graph mode.
    traceback: (list of trace frames) the stack trace of the op's creation.
      Available only under graph model.
    stack_height_limit: (int or None) If int, limit to the height of the stack
      trace printed in the error message. If None, no limit to the height.
    path_length_limit: (int or None) Length limit for file paths included in the
      formatted stack trace.

  Returns:
    (str) A formatted error message.
  """
  eager_vs_graph_qualifier = "graph" if graph else "eagerly-executing"
  message = "\n"
  message += (
      "\n!!! Detected Infinity or NaN in output %d of "
      "%s op \"%s\" (# of outputs: %d) !!!\n" %
      (slot, eager_vs_graph_qualifier, op_type, num_outputs))

  message += "  dtype: %s\n" % tensor.dtype
  message += "  shape: %s\n" % (tensor.shape,)

  if not graph:
    # This is an eager tensor. We can get its numpy value and count
    # NaNs and Infs.
    is_inf = np.isinf(tensor)

    num_neg_inf = np.sum(np.logical_and(np.less(tensor, 0.), is_inf))
    num_pos_inf = np.sum(np.logical_and(np.greater(tensor, 0.), is_inf))
    num_nan = np.sum(np.isnan(tensor))
    if num_neg_inf > 0:
      message += "  # of -Inf elements: %s\n" % num_neg_inf
    if num_pos_inf > 0:
      message += "  # of +Inf elements: %s\n" % num_pos_inf
    if num_nan:
      message += "  # of +NaN elements: %s\n" % num_nan

  if len(inputs) > 1:
    message += "\n  Input tensors (%d):\n" % len(inputs)
    for slot, input_tensor in enumerate(inputs):
      message += "         %d: %s\n" % (
          slot, _maybe_lookup_original_input_tensor(graph, input_tensor))
  elif len(inputs) == 1:
    message += "\n  Input tensor: %s\n" % (
        _maybe_lookup_original_input_tensor(graph, inputs[0]))
  if graph and graph.name:
    message += "  Graph name: \"%s\"\n" % graph.name

  # Format the stack trace for the op's creation. We omit files that
  # belong to tensorflow itself.
  if graph and traceback:
    message += (
        "\n  Stack trace of op's creation (\"->\": inferred user code):\n")
    if stack_height_limit is not None and len(traceback) > stack_height_limit:
      num_omitted_frames = len(traceback) - stack_height_limit
      message += "    + ... (Omitted %d frames)\n" % num_omitted_frames
    for filepath, lineno, function_name, source_line in traceback[
        -stack_height_limit:]:
      user_code_indicator = "    "
      if not source_utils.guess_is_tensorflow_py_library(filepath):
        user_code_indicator = " -> "

      message += "    + %s (L%d) %s\n" % (
          limit_string_length(filepath, path_length_limit), lineno,
          function_name)
      if source_line is not None:
        message += "%s|   %s\n" % (user_code_indicator, source_line)
  message += "\n"
  return message


def _check_numerics_callback(stack_height_limit,
                             path_length_limit,
                             op_type,
                             inputs,
                             attrs,
                             outputs,
                             op_name=None,
                             graph=None):
  """Eager-function unified callback for checking numerics."""
  del attrs, op_name  # Unused

  if compat.as_bytes(op_type) in _CHECK_NUMERICS_CALLBACK_SKIP_OPS:
    return
  if graph:
    # Under graph mode. Insert check_numerics op.
    instrumented_outputs = []
    for slot, output in enumerate(outputs):
      if output.dtype.is_floating:
        checked_output = array_ops.check_numerics(
            output,
            get_check_numerics_error_message(
                slot, len(outputs), op_type, output, inputs,
                graph=graph, traceback=output.op.traceback))
        if graph.name:
          _CHECK_NUMERICS_INPUT_LOOKUP[
              graph.name][checked_output.name] = output
        instrumented_outputs.append(checked_output)
      else:
        instrumented_outputs.append(output)
    return instrumented_outputs
  else:
    if compat.as_bytes(op_type) == b"CheckNumerics":
      # TODO(b/140334369): Remove this special casing logic once op_callback.
      # automatically prevents infinite recursion in eager mode.
      return
    # Under eager mode. Eagerly execute check_numerics op.
    for slot, output in enumerate(outputs):
      if output.dtype.is_floating:
        array_ops.check_numerics(
            output,
            get_check_numerics_error_message(
                slot, len(outputs), op_type, output, inputs,
                stack_height_limit=stack_height_limit,
                path_length_limit=path_length_limit))


def check_numerics(stack_height_limit=30,
                   path_length_limit=50):
  r"""Creates a context manager that checks numerics of tensors in ops.

  This context manager works for eagerly-executed ops and ops executed in
  `tf.function`s (graphs) in a unified way.

  When a op's float-type output tensor contains any Infinity or NaN, an
  `tf.errors.InvalidArgumentError` will be thrown, with an error message that
  reveals the following information:
    - The type of the op that generated the tensor with bad numerics.
    - Data type (dtype) of the tensor.
    - Shape of the tensor (to the extent known at the time of eager execution
      or graph construction).
    - (Graph mode only): Name of the containing graph.
    - (Graph mode only): The stack trace of the intra-graph op's creation,
      with a stack-height limit and a path-length limit for visual clarity.
      The stack frames that belong to the user's code (as opposed to
      tensorflow's internal code) are highlighted with a text arrow ("->").
    - (Eager mode only): How many of the offending tensor's elements are
      `Infinity` and `NaN`, respectively.

  Args:
    stack_height_limit: Limit to the height of the printed stack trace.
      Applicable only to ops in `tf.function`s (graphs).
    path_length_limit: Limit to the file path included in the printed stack
      trace. Applicable only to ops in `tf.function`s (graphs).

  Returns:
    A thread-local context manager.
  """
  # TODO(cais): Once this is exposed as a public API add code example in the
  # doc string above.

  return op_callbacks.op_callback(functools.partial(
      _check_numerics_callback,
      stack_height_limit,
      path_length_limit))
