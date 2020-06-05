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

from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import logging_ops
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


class TpuOutsideCompilationTest(test.TestCase):

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

  def testOutsideCompilationControlFlowIf(self):
    strategy = get_tpu_strategy()

    def outside_fn(x):
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    @def_function.function
    def train_step():

      def tpu_fn(x):
        x2 = x + 5.0
        if x < 50.0:
          return tpu.outside_compilation(outside_fn, x2)
        else:
          return x2

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(36., shape=(strategy.num_replicas_in_sync)))

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


if __name__ == "__main__":
  test.main()
