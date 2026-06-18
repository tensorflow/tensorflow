# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for summary op transformations."""
import os
import os.path

from absl import flags

from tensorflow.core.function.runtime_client import runtime_client
from tensorflow.core.util import event_pb2
from tensorflow.python.data.ops import readers
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

FLAGS = flags.FLAGS


class SummaryOpsTransformationTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self.summary_dir = os.path.join(FLAGS.test_tmpdir, 'mylogs')

    # Clean up any summary directories before starting the test so we can
    # validate that summaries are only written when enabled.
    try:
      gfile.DeleteRecursively(self.summary_dir)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  @test_util.run_v2_only
  def test_strip_summary_ops(self):
    def normalize_while_node(fndef):
      """Helper method to normalize the while node for comparison."""
      for node in fndef.node_def:
        if node.op == 'While':
          # The names of the nested functions are expected to be different
          # because they will have a uid appended to them.
          node.attr['body'].func.name = 'while_body'
          node.attr['cond'].func.name = 'while_cond'

          # The summary_writer and `include_summary` args are expected to be
          # passed in and out of the transformed function as we do not modify
          # the function signatures.
          # Expect a mismatch in input and output types/shapes.
          node.attr['T'].ClearField('list')
          node.attr['output_shapes'].ClearField('list')
          expected_inputs = {
              'write_summary_summary_cond_input_1',
              'record_summary',
          }
          if 'record_summary' not in node.input:
            continue
          inputs = node.input
          node.ClearField('input')
          node.input.extend(inp for inp in inputs if inp not in expected_inputs)
          node.attr['_num_original_outputs'].i -= 2

      return fndef

    def normalize_fdef(fndef):
      """Method to normalize the tf.function's FunctionDefs for comparison."""
      # Normalize the names for comparison as they have a uid appended.
      fndef.signature.name = '__inference_add'

      # The summary writer is expected to be passed into the transformed fn.
      inputs = fndef.signature.input_arg
      fndef.signature.ClearField('input_arg')
      fndef.signature.input_arg.extend(
          inp
          for inp in inputs
          if inp.name != 'write_summary_summary_cond_input_1'
      )

      # The disable_summaries_at_runtime attr is expected to be cleared.
      fndef.attr['disable_summaries_at_runtime'].ClearField('list')
      return fndef

    writer = summary_ops_v2.create_file_writer_v2(self.summary_dir)
    var = variables.Variable(1.0)

    def remove_writer_attr(fndef):
      arg_attr = fndef.arg_attr
      attr_idx = None
      # tf.function uses TraceType to create placeholder for captures.
      # An extra "_user_specified_name" attr will be added to the placeholder.
      for idx in arg_attr:
        if arg_attr[idx].attr['_user_specified_name'].s == b'input_1':
          attr_idx = idx
          break
      if attr_idx is not None:
        # Copy subsequent arg_attr to ensure indexes are continuous
        for idx in range(attr_idx, len(arg_attr) - 1):
          fndef.arg_attr[idx].CopyFrom(fndef.arg_attr[idx + 1])
        del fndef.arg_attr[len(arg_attr) - 1]
      return fndef

    @polymorphic_function.function(
        autograph=False,
        experimental_attributes={
            'disable_summaries_at_runtime': ['record_summary', False]
        },
    )
    def add(x, y, record_summary, include_summary):
      def body(step, result):
        result += math_ops.cast(step, dtypes.float32)
        var.assign(result)
        if include_summary:
          # Perform a summary write in a nested function.
          with writer.as_default():
            summary_ops_v2.set_step(step)
            summary_ops_v2.write('my_metric', result, step=step)
          writer.flush()
        return math_ops.add(step, 1)

      result = math_ops.add(x, y)
      step = constant_op.constant(0, dtypes.int64)

      with summary_ops_v2.record_if(record_summary):
        if include_summary:
          # Perform a summary write in the main function body.
          with writer.as_default():
            summary_ops_v2.set_step(step)
            summary_ops_v2.write('my_metric', result, step=step)
          writer.flush()

        step = math_ops.add(step, 1)
        loop_cond = lambda i: math_ops.less(i, 3)
        loop_body = lambda i: body(i, result)
        step = while_loop.while_loop_v2(loop_cond, loop_body, [step])
        var.assign(result)
      return result

    one = constant_op.constant(1.0, dtypes.float32)
    inputs_with_summaries = [one, one, constant_op.constant(True), True]
    inputs_without_summaries = [one, one, constant_op.constant(False), False]
    inputs_without_summaries_at_runtime = [
        one,
        one,
        constant_op.constant(False),
        True,
    ]

    # Ensure the result of `add` is the same with and without summaries.
    self.assertEqual(
        add(*inputs_with_summaries), add(*inputs_without_summaries)
    )

    # Ensure the result of `add` is the same when summaries have been stripped
    # at trace time.
    self.assertEqual(
        add(*inputs_without_summaries_at_runtime),
        add(*inputs_without_summaries),
    )

    # Force a trace of `add` where summaries have been stripped at trace time.
    expected = add.get_concrete_function(*inputs_without_summaries).function_def

    # Extract the trace of `add` where summaries have been stripped in the
    # runtime.
    function_name = add.get_concrete_function(
        *inputs_without_summaries_at_runtime
    ).function_def.signature.name
    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fndef = rt.GetFunctionProto(function_name + '__instance__no_summaries')

    # Normalize the fndefs and compare them for equivalence.
    fndef = normalize_fdef(normalize_while_node(fndef))
    fndef = remove_writer_attr(fndef)
    expected = normalize_fdef(normalize_while_node(expected))
    self.assertProtoEquals(expected, fndef)

    # Verify that summaries were only written when executing with the
    # `inputs_with_summaries` argument.
    num_summary_events = 0
    summary_files = [
        os.path.join(self.summary_dir, sf)
        for sf in gfile.ListDirectory(self.summary_dir)
    ]
    for record in readers.TFRecordDatasetV2(
        filenames=summary_files
    ).as_numpy_iterator():
      event = event_pb2.Event()
      event.ParseFromString(record)
      if event.HasField('summary'):
        num_summary_events += 1
    # 3 Events are written by `add` when summaries are enabled.
    self.assertEqual(num_summary_events, 3)


if __name__ == '__main__':
  test.main()
