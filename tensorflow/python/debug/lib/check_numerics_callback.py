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
import threading

import numpy as np

from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import op_callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


# Many ops have benign NaN outputs, and running them with check_numerics
# on will create unwanted errors
# TODO(b/142497024): Replace this whitelist with function decorators in the ops
IGNORE_OP_OUTPUTS = (
    # For FusedBatchNorm, if the input tensor is empty then batch_mean and
    # batch_variance will be NaN. reserve_space holds intermediate values
    # derived from batch_mean and batch_variance used for gradient calculation
    (b"FusedBatchNorm", 1),  # batch_mean
    (b"FusedBatchNorm", 2),  # batch_variance
    (b"FusedBatchNorm", 3),  # reserve_space_1
    (b"FusedBatchNorm", 4),  # reserve_space_2

    # Same as above
    (b"FusedBatchNormV2", 1),  # batch_mean
    (b"FusedBatchNormV2", 2),  # batch_variance
    (b"FusedBatchNormV2", 3),  # reserve_space_1
    (b"FusedBatchNormV2", 4),  # reserve_space_2

    # Same as above, but reserve_space_3 holds additional intermediate values
    (b"FusedBatchNormV3", 1),  # batch_mean
    (b"FusedBatchNormV3", 2),  # batch_variance
    (b"FusedBatchNormV3", 3),  # reserve_space_1
    (b"FusedBatchNormV3", 4),  # reserve_space_2
    (b"FusedBatchNormV3", 5),  # reserve_space_3
)


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


# A dictionary that supports looking up the original input tensor names.
_CHECK_NUMERICS_INPUT_LOOKUP = collections.defaultdict(dict)


def _maybe_lookup_original_input_tensor(graph, tensor):
  if (graph and
      graph in _CHECK_NUMERICS_INPUT_LOOKUP and
      tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph]):
    return _CHECK_NUMERICS_INPUT_LOOKUP[graph][tensor.name]
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
  if graph and hasattr(graph, "name") and graph.name:
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


def _check_numerics_callback(op_type,
                             inputs,
                             attrs,
                             outputs,
                             op_name=None,
                             graph=None):
  """Eager-function unified callback for checking numerics."""
  del attrs, op_name  # Unused
  op_type_bytes = compat.as_bytes(op_type)
  if op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
    return
  if graph:
    # Under graph mode. Insert check_numerics op.
    instrumented_outputs = []
    for slot, output in enumerate(outputs):
      if (output.dtype.is_floating and
          (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
        checked_output = array_ops.check_numerics(
            output,
            get_check_numerics_error_message(
                slot, len(outputs), op_type, output, inputs,
                graph=graph, traceback=output.op.traceback))
        _CHECK_NUMERICS_INPUT_LOOKUP[graph][checked_output.name] = output
        instrumented_outputs.append(checked_output)
      else:
        instrumented_outputs.append(output)
    return instrumented_outputs
  else:
    if op_type_bytes == b"CheckNumerics":
      # TODO(b/140334369): Remove this special casing logic once op_callback.
      # automatically prevents infinite recursion in eager mode.
      return
    # Under eager mode. Eagerly execute check_numerics op.
    for slot, output in enumerate(outputs):
      if (output.dtype.is_floating and
          (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS):
        array_ops.check_numerics(
            output,
            get_check_numerics_error_message(
                slot, len(outputs), op_type, output, inputs,
                stack_height_limit=_state.config.stack_height_limit,
                path_length_limit=_state.config.path_length_limit))


CheckNumericsConfig = collections.namedtuple(
    "CheckNumericsConfig", "stack_height_limit path_length_limit")
_state = threading.local()


@tf_export("debugging.enable_check_numerics")
def enable_check_numerics(stack_height_limit=30,
                          path_length_limit=50):
  r"""Enable tensor numerics checking in an eager/graph unified fashion.

  The numerics checking mechanism will cause any TensorFlow eager execution or
  graph execution to error out as soon as an op's output tensor contains
  infinity or NaN.

  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.

  This method takes effect only on the thread in which it is called.

  When a op's float-type output tensor contains any Infinity or NaN, an
  `tf.errors.InvalidArgumentError` will be thrown, with an error message that
  reveals the following information:
    - The type of the op that generated the tensor with bad numerics.
    - Data type (dtype) of the tensor.
    - Shape of the tensor (to the extent known at the time of eager execution
      or graph construction).
    - Name of the containing graph (if available).
    - (Graph mode only): The stack trace of the intra-graph op's creation,
      with a stack-height limit and a path-length limit for visual clarity.
      The stack frames that belong to the user's code (as opposed to
      tensorflow's internal code) are highlighted with a text arrow ("->").
    - (Eager mode only): How many of the offending tensor's elements are
      `Infinity` and `NaN`, respectively.

  Once enabled, the check-numerics mechanism can be disabled by using
  `tf.debugging.disable_check_numerics()`.

  Example usage:

  1. Catching infinity during the execution of a `tf.function` graph:

     ```py
     import tensorflow as tf

     tf.debugging.enable_check_numerics()

     @tf.function
     def square_log_x_plus_1(x):
       v = tf.math.log(x + 1)
       return tf.math.square(v)

     x = -1.0

     # When the following line runs, a function graph will be compiled
     # from the Python function `log_x_plus_1()`. Due to the
     # `enable_check_numerics()` call above, the graph will contain
     # numerics checking ops that will run during the function graph's
     # execution. The function call generates an -infinity when the Log
     # (logarithm) op operates on the output tensor of the Add op.
     # The program errors out at this line, printing an error message.
     y = log_x_plus_1(x)
     z = -y
    ```

  2. Catching NaN during eager execution:

     ```py
     import numpy as np
     import tensorflow as tf

     tf.debugging.enable_check_numerics()

     x = np.array([[0.0, -1.0], [4.0, 3.0]])

     # The following line executes the Sqrt op eagerly. Due to the negative
     # element in the input array, a NaN is generated. Due to the
     # `enable_check_numerics()` call above, the program errors immediately
     # at this line, printing an error message.
     y = tf.math.sqrt(x)
     z = tf.matmul(y, y)
     ```

  Args:
    stack_height_limit: Limit to the height of the printed stack trace.
      Applicable only to ops in `tf.function`s (graphs).
    path_length_limit: Limit to the file path included in the printed stack
      trace. Applicable only to ops in `tf.function`s (graphs).
  """

  if not hasattr(_state, "config"):
    _state.config = CheckNumericsConfig(
        stack_height_limit=stack_height_limit,
        path_length_limit=path_length_limit)
  op_callbacks.add_op_callback(_check_numerics_callback)

  logging.info(
      "Enabled check-numerics callback in thread %s",
      threading.current_thread().name)


@tf_export("debugging.disable_check_numerics")
def disable_check_numerics():
  """Disable the eager/graph unified numerics checking mechanism.

  This method can be used after a call to `tf.debugging.enable_check_numerics()`
  to disable the numerics-checking mechanism that catches inifnity and NaN
  values output by ops executed eagerly or in tf.function-compiled graphs.

  This method is idempotent. Calling it multiple times has the same effect
  as calling it once.

  This method takes effect only on the thread in which it is called.
  """
  try:
    op_callbacks.remove_op_callback(_check_numerics_callback)
    logging.info(
        "Disabled check-numerics callback in thread %s",
        threading.current_thread().name)
  except KeyError:
    # Tolerate disabling the check numerics callback without
    # enable_check_numerics() being called first.
    pass
