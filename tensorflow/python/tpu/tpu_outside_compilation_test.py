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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_strategy_util

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
  return tpu_lib.TPUStrategy(resolver)


class TpuOutsideCompilationTest(test.TestCase, parameterized.TestCase):

  def testResourceVariableAssignOnHost(self):
    strategy = get_tpu_strategy()
    with strategy.scope():
      v = variables.Variable(
          0.0, aggregation=variables.VariableAggregation.MEAN)
    v2 = variables.Variable(0.0, aggregation=variables.VariableAggregation.MEAN)

    def assign_fn():
      v2.assign_add(4.0)

    @def_function.function
    def train_step():

      def assign_add():
        v.assign_add(2.0)
        tpu.outside_compilation(assign_fn)
        v.assign_add(3.0)

      strategy.run(assign_add)
      return

    train_step()
    self.assertAllEqual(4.0 * strategy.num_replicas_in_sync, v2.numpy())
    self.assertAllEqual(5.0, v.numpy())

  def testHostNoInput(self):
    strategy = get_tpu_strategy()

    def outside_fn():
      logging_ops.print_v2("Outside compiled")

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        tpu.outside_compilation(outside_fn)
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
        tpu.outside_compilation(outside_fn, x2)
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
        output = tpu.outside_compilation(outside_fn, x2)
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
        c, d, e = tpu.outside_compilation(outside_fn, a, b)
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
        output1 = tpu.outside_compilation(outside_fn1, x2)
        x3 = output1 + 3.0
        output2 = tpu.outside_compilation(outside_fn2, x3)
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
          return tpu.outside_compilation(outside_fn, x2)
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
          x2 = tpu.outside_compilation(outside_fn, x2)
        return x2 + 4.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(58., shape=(strategy.num_replicas_in_sync)))

  def testSummary(self):
    strategy = get_tpu_strategy()

    def host_computation(x):
      summary.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        x = x + 1.0
        y = tpu.outside_compilation(host_computation, x)
        y = tpu.outside_compilation(host_computation, x)
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
      summary.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step(take_true_branch):

      def computation(x):
        x = x + 1.0
        if x < 5.0:
          y = tpu.outside_compilation(host_computation, x)
          y = tpu.outside_compilation(host_computation, x)
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
      summary.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():

      def computation(x):
        n = 0
        while n < 3:
          x = x + 1.0
          y = tpu.outside_compilation(host_computation, x)
          y = tpu.outside_compilation(host_computation, x)
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


if __name__ == "__main__":
  test.main()
