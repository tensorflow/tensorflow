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
"""Tests for TPU outside compilation."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorboard.plugins.histogram import summary_v2 as histogram_summary_v2
from tensorboard.plugins.image import summary_v2 as image_summary_v2
from tensorboard.plugins.scalar import summary_v2 as scalar_summary_v2
from tensorflow.core.util import event_pb2
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu.ops import tpu_ops

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_strategy_util.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategyV2(resolver)


def computation_with_string_ops(x):
  output = string_ops.string_format("1{}", x)
  return string_ops.string_to_number(output)


def _events_from_logdir(test_case, logdir):
  """Reads summary events from log directory."""
  test_case.assertTrue(gfile.Exists(logdir))
  files = gfile.ListDirectory(logdir)
  test_case.assertLen(files, 1)
  records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def _rewrite_func_wrapper(tf_func):

  def tpu_fn(*args, **kwargs):
    # tpu.rewrite only accepts list of tensors as input. We need to flatten
    # keyword arguments to meet this requirement.
    concrete = tf_func.get_concrete_function(*(list(args) +
                                               list(kwargs.values())))
    return tpu.rewrite(concrete.__call__, list(args) + list(kwargs.values()))

  return def_function.function(tpu_fn)


def _tpu_partitioned_call_wrapper(tf_func):
  """Wrap a tensorflow Function with TPUPartitionedCall."""

  def inner_func(*args, **kwargs):
    concrete = tf_func.get_concrete_function(*args, **kwargs)
    # TPUPartitionedCall only accepts list of tensors as input args.
    # Flatten keyword arguments and do some basic ordering:
    # Positional args + Flattened keyword args + Captured args.
    op_args = list(args) + list(kwargs.values()) + concrete.captured_inputs
    return tpu_functional.TPUPartitionedCall(
        args=op_args,
        device_ordinal=tpu_ops.tpu_ordinal_selector(),
        Tout=[o.type for o in concrete.function_def.signature.output_arg],
        f=concrete)

  return def_function.function(inner_func)


class TpuOutsideCompilationTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TpuOutsideCompilationTest, self).setUp()
    config.set_soft_device_placement(False)

  def testHostNoInput(self):
    strategy = get_tpu_strategy()

    def outside_fn():
      logging_ops.print_v2("Outside compiled")

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testHostInputOnly(self):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testJitCompile(self):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)

    # jit_compile=True should have no effect for TPU.
    @def_function.function(jit_compile=True)
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testHostInputOutput(self):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        output = tpu_replication.outside_compilation(outside_fn, x2)
        return output

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(36., shape=(strategy.num_replicas_in_sync)))

  def testHostMultipleInputs(self):
    strategy = get_tpu_strategy()
    val0 = np.arange(6).reshape((2, 3)).astype(np.float32)
    val1 = np.arange(6).reshape((3, 2)).astype(np.float32)

    def outside_fn(arg0, arg1):
      tmp = array_ops.reshape(arg1, array_ops.shape(arg0))
      ret0 = arg0 + tmp
      ret1 = math_ops.matmul(arg0, arg1)
      ret2 = array_ops.concat([arg0, tmp], 0)
      return ret0, ret1, ret2

    @def_function.function
    def train_step():

      def tpu_fn(x, y):
        a = x + 7.0
        b = y * 2.0
        c, d, e = tpu_replication.outside_compilation(outside_fn, a, b)
        return (math_ops.reduce_max(c) + math_ops.reduce_min(d) +
                math_ops.reduce_sum(e))

      return strategy.run(tpu_fn, args=(val0, val1))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(213., shape=(strategy.num_replicas_in_sync)))

  def testMultipleClusters(self):
    strategy = get_tpu_strategy()

    def outside_fn1(x):
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    def outside_fn2(x):
      logging_ops.print_v2("Outside compiled", x)
      return x - 18.0

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        output1 = tpu_replication.outside_compilation(outside_fn1, x2)
        x3 = output1 + 3.0
        output2 = tpu_replication.outside_compilation(outside_fn2, x3)
        return output2

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(21., shape=(strategy.num_replicas_in_sync)))

  @parameterized.parameters((True), (False))
  def testOutsideCompilationControlFlowIf(self, take_true_branch):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    input_value = 51.0 if take_true_branch else 25.0

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        if x < 50.0:
          return tpu_replication.outside_compilation(outside_fn, x2)
        else:
          return x2

      return strategy.run(tpu_fn, args=(input_value,))

    output_value = 36.0
    if take_true_branch:
      output_value = 56.0
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(
            output_value, shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationControlFlowWhile(self):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        while x2 < 50.0:
          x2 = tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 4.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(58., shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationHostControlFlow(self):
    """Tests that control flow on host for outside_compilation works."""
    strategy = get_tpu_strategy()

    def outside_fn(x):
      n = 0
      while n < 4:
        x = x + 6.0
        n = n + 1
      return x

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        x2 = tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 4.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(58., shape=(strategy.num_replicas_in_sync)))

  def testSummary(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        x = x + 1.0
        y = tpu_replication.outside_compilation(host_computation, x)
        y = tpu_replication.outside_compilation(host_computation, x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))

  @parameterized.parameters((True), (False))
  def testSummaryInCond(self, take_true_branch):
    strategy = get_tpu_strategy()

    def host_computation(x):
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step(take_true_branch):

      def computation(x):
        x = x + 1.0
        if x < 5.0:
          y = tpu_replication.outside_compilation(host_computation, x)
          y = tpu_replication.outside_compilation(host_computation, x)
          x = y
        return x + 1.0

      if take_true_branch:
        return strategy.run(computation, args=(2.0,))
      else:
        return strategy.run(computation, args=(10.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)

    output_value = 12.
    if take_true_branch:
      output_value = 7.
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step(take_true_branch)),
          constant_op.constant(
              output_value, shape=(strategy.num_replicas_in_sync)))

  def testSummaryInWhile(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        n = 0
        while n < 3:
          x = x + 1.0
          y = tpu_replication.outside_compilation(host_computation, x)
          y = tpu_replication.outside_compilation(host_computation, x)
          x = y
          n = n + 1
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(31., shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationAtHeadAndTail(self):
    """Tests that outside_compilation at head/tail of TPU computation works."""
    strategy = get_tpu_strategy()

    def host_computation(x):
      return x * 2.0

    @def_function.function
    def train_step():

      def computation(x):
        w = tpu_replication.outside_compilation(host_computation, x)
        y = w + 1.0
        z = tpu_replication.outside_compilation(host_computation, y)
        return z + 5.0

      return strategy.run(computation, args=(2.0,))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(15., shape=(strategy.num_replicas_in_sync)))

  def testGradientAcrossOutsideCompilation(self):
    """Tests compiled gradients can contain host computations."""
    strategy = get_tpu_strategy()

    def host_computation(a):
      b = a * a
      c = b * b
      return c

    @def_function.function
    def train_step():
      def computation(x, y):
        a = x + 7.0
        b = tpu_replication.outside_compilation(host_computation, a)
        c = b * y
        d = gradients_impl.gradients(
            [c], [x], colocate_gradients_with_ops=True)[0]
        return d

      return strategy.run(computation, args=(2.0, 3.0))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(8748., shape=(strategy.num_replicas_in_sync)))

  def testGradientOfGradientAcrossOutsideCompilation(self):
    """Tests compiled gradients of gradients can contain host computations."""
    strategy = get_tpu_strategy()

    def host_computation(a):
      b = a * a
      c = b * b
      return c

    @def_function.function
    def train_step():
      def computation(x, y):
        a = x + 7.0
        b = tpu_replication.outside_compilation(host_computation, a)
        c = b * y
        d = gradients_impl.gradients(
            [c], [x], colocate_gradients_with_ops=True)[0]
        e = gradients_impl.gradients(
            [d], [x], colocate_gradients_with_ops=True)[0]
        return e

      return strategy.run(computation, args=(2.0, 3.0))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(2916., shape=(strategy.num_replicas_in_sync)))

  def testColocateGradientWithOutsideCompiledOp(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step():

      @def_function.function
      def tpu_fn(x):
        x1 = tpu_replication.outside_compilation(math_ops.sqrt, x)
        grad = gradients_impl.gradients([x1], [x],
                                        colocate_gradients_with_ops=True)[0]
        sqrt = [
            op for op in ops.get_default_graph().get_operations()
            if op.type == "Sqrt"
        ][0]
        sqrt_grad = [
            op for op in ops.get_default_graph().get_operations()
            if op.type == "SqrtGrad"
        ][0]
        assert sqrt.get_attr(
            tpu_replication._OUTSIDE_COMPILATION_ATTR) == b"0"
        assert (sqrt_grad.get_attr(
            tpu_replication._OUTSIDE_COMPILATION_ATTR) == b"0.gradients/uid"
               )
        return grad

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(.1, shape=(strategy.num_replicas_in_sync)))


class OutsideCompilationOnUnsupportedOpTest(test.TestCase,
                                            parameterized.TestCase):

  def setUp(self):
    super(OutsideCompilationOnUnsupportedOpTest, self).setUp()
    config.set_soft_device_placement(True)

  def testStringOpWithManualOutsideCompilation(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(x):

      def computation(x):
        return tpu_replication.outside_compilation(
            computation_with_string_ops, x)

      return strategy.run(computation, args=(x,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  def testStringOpWithAutoOutsideCompilation(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(x):

      def computation(x):
        return computation_with_string_ops(x)

      return strategy.run(computation, args=(x,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  # Regression test for b/180509859.
  def testImageSummary(self):
    strategy = get_tpu_strategy()

    def run():

      @def_function.function
      def sample_sequence():
        bsz = 3
        max_length = 32 * 32

        def f():

          def body(step, tokens):
            next_token = random_ops.random_uniform([bsz])
            tokens = tokens.write(step, next_token)
            return (step + 1, tokens)

          def cond(step, tokens):
            del tokens
            return math_ops.less(step, max_length)

          tokens_var = tensor_array_ops.TensorArray(
              dtype=dtypes.float32,
              size=max_length,
              dynamic_size=False,
              clear_after_read=False,
              element_shape=(bsz,),
              name="tokens_accumulator",
          )

          step = constant_op.constant(0)
          step, tokens_var = while_loop.while_loop(cond, body,
                                                   [step, tokens_var])

          image_flat = array_ops.transpose(tokens_var.stack(), [1, 0])
          image = array_ops.tile(
              array_ops.reshape(image_flat, [bsz, 32, 32, 1]), [1, 1, 1, 3])
          image_summary_v2.image("image_sample", image,
                                 constant_op.constant(5, dtype=dtypes.int64))

        return strategy.run(f)

      sample_sequence()

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      run()
    events = _events_from_logdir(self, logdir)
    decoded_image = image_ops.decode_png(
        events[1].summary.value[0].tensor.string_val[2]).numpy()
    # Ensure that non-zero values were written to the image summary.
    self.assertNotAllEqual(
        array_ops.zeros((3072,), dtype=dtypes.float32),
        list(decoded_image.flat))

  def testSummaryWithAutoOutsideCompilation(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  def testNestedFunctionScalarSummary(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      @def_function.function
      def computation(x):
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  def testHistogramSummaryWithAutoOutsideCompilation(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      histogram_summary_v2.histogram("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  @parameterized.parameters((True), (False))
  def testSummaryControlFlowIfWithAutoOutsideCompilation(
      self, take_true_branch):
    strategy = get_tpu_strategy()

    @def_function.function
    def step():

      def computation(x):
        x = x + 1.0
        if x < 5:
          scalar_summary_v2.scalar("x", x, step=0)
          x = x * 2.0
        return x + 1.0

      if take_true_branch:
        return strategy.run(computation, args=(2.0,))
      else:
        return strategy.run(computation, args=(10.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    output_value = 12.
    if take_true_branch:
      output_value = 7.
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(
              output_value, shape=(strategy.num_replicas_in_sync)))
    if take_true_branch:
      events = _events_from_logdir(self, logdir)
      # There will be 2 entries: 1 summary file header entry, and 1 entry
      # written by host.
      #
      self.assertLen(events, 2)
      self.assertEqual(events[1].summary.value[0].tag, "cond/x")

  def testAutoOutsideCompilationWithFunctionalNodes(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(a, b):

      def fn(a, b):
        fn1 = lambda: computation_with_string_ops(a * 100)
        fn2 = lambda: computation_with_string_ops(a)
        pred = math_ops.greater_equal(a, b)
        result = array_ops.identity(
            control_flow_ops.cond(pred, fn1, fn2),
            name="uncompilable_control_flow")
        return result

      return strategy.run(fn, args=(a, b))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0.0, -1.0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  def testRandomOpsWithAutoOutsideCompilation(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step():

      def computation():
        return random_ops.random_normal(shape=[1, 2, 3])

      return strategy.run(computation, args=())

    self.assertAllEqual(
        strategy.experimental_local_results(train_step())[0].shape, [1, 2, 3])

  def testOutsideCompilationWithTPUPartitionedCallOp(self):
    """Tests that control flow with TPUPartitionedCall including outside_compilation works."""
    get_tpu_strategy()

    def host_computation(x):
      return x + 1

    @def_function.function()
    def train_step(x):
      x2 = x + 5.0
      logging_ops.print_v2(x2)
      x2 = tpu_replication.outside_compilation(host_computation, x2)
      return x2 + 4.0

    tpu_fn = _rewrite_func_wrapper(train_step)
    partitioned_tpu_fn = _tpu_partitioned_call_wrapper(tpu_fn)

    concrete = partitioned_tpu_fn.get_concrete_function(
        x=tensor_spec.TensorSpec(
            shape=(1), dtype=dtypes.float32, name="input_tensor"))

    self.assertIsInstance(
        concrete(array_ops.ones((1), dtype=dtypes.float32))[0], ops.Tensor)


if __name__ == "__main__":
  test.main()
