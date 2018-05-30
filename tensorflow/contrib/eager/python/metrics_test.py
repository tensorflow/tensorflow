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
"""Tests for Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.contrib.eager.python import metrics
from tensorflow.contrib.summary import summary_test_util
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpointable import util as checkpointable_utils


class MetricsTest(test.TestCase):

  def testMean(self):
    m = metrics.Mean()
    m([1, 10, 100])
    m(1000)
    m([10000.0, 100000.0])
    self.assertEqual(111111.0/6, m.result().numpy())
    self.assertEqual(dtypes.float64, m.dtype)
    self.assertEqual(dtypes.float64, m.result().dtype)

  def testVariableCollections(self):
    with context.graph_mode(), ops.Graph().as_default():
      m = metrics.Mean()
      m(1000)
      self.assertEqual(
          set(m.variables),
          set(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES)))
      self.assertEqual(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), [])
      self.assertEqual(
          set(m.variables),
          set(ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

  def testUseGlobalVariablesCollections(self):
    with context.graph_mode(), ops.Graph().as_default():
      m = metrics.Mean(use_global_variables=True)
      m(1000)
      self.assertEqual(
          set(m.variables),
          set(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      self.assertEqual(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES), [])
      self.assertEqual(
          set(m.variables),
          set(ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))

  def testInitVariables(self):
    m = metrics.Mean()
    m([1, 10, 100, 1000])
    m([10000.0, 100000.0])
    self.assertEqual(111111.0/6, m.result().numpy())
    m.init_variables()
    m(7)
    self.assertEqual(7.0, m.result().numpy())

  def testWriteSummaries(self):
    m = metrics.Mean()
    m([1, 10, 100])
    training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logdir, max_queue=0,
        name="t0").as_default(), summary_ops.always_record_summaries():
      m.result()  # As a side-effect will write summaries.

    events = summary_test_util.events_from_logdir(logdir)
    self.assertEqual(len(events), 2)
    self.assertEqual(events[1].summary.value[0].simple_value, 37.0)

  def testWeightedMean(self):
    m = metrics.Mean()
    m([1, 100, 100000], weights=[1, 0.2, 0.3])
    m([500000, 5000, 500])  # weights of 1 each
    self.assertNear(535521/4.5, m.result().numpy(), 0.001)

  def testMeanDtype(self):
    # Can override default dtype of float64.
    m = metrics.Mean(dtype=dtypes.float32)
    m([0, 2])
    self.assertEqual(1, m.result().numpy())
    self.assertEqual(dtypes.float32, m.dtype)
    self.assertEqual(dtypes.float32, m.result().dtype)

  def testAccuracy(self):
    m = metrics.Accuracy()
    m([0, 1, 2, 3], [0, 0, 0, 0])  # 1 correct
    m([4], [4])  # 1 correct
    m([5], [0])  # 0 correct
    m([6], [6])  # 1 correct
    m([7], [2])  # 0 correct
    self.assertEqual(3.0/8, m.result().numpy())
    self.assertEqual(dtypes.float64, m.dtype)
    self.assertEqual(dtypes.float64, m.result().dtype)

  def testAccuracyDifferentShapes(self):
    m = metrics.Accuracy()
    with self.assertRaises(errors.InvalidArgumentError):
      m([[0], [0]], [0, 1])

  def testWeightedAccuracy(self):
    m = metrics.Accuracy()
    # 1 correct, total weight of 2
    m([0, 1, 2, 3], [0, 0, 0, 0], weights=[1, 1, 0, 0])
    m([4], [4], weights=[0.5])  # 1 correct with a weight of 0.5
    m([5], [0], weights=[0.5])  # 0 correct, weight 0.5
    m([6], [6])  # 1 correct, weight 1
    m([7], [2])  # 0 correct, weight 1
    self.assertEqual(2.5/5, m.result().numpy())

  def testAccuracyDtype(self):
    # Can override default dtype of float64.
    m = metrics.Accuracy(dtype=dtypes.float32)
    m([0, 0], [0, 1])
    self.assertEqual(0.5, m.result().numpy())
    self.assertEqual(dtypes.float32, m.dtype)
    self.assertEqual(dtypes.float32, m.result().dtype)

  def testTwoMeans(self):
    # Verify two metrics with the same class and name don't
    # accidentally share state.
    m1 = metrics.Mean()
    m1(0)
    m2 = metrics.Mean()
    m2(2)
    self.assertAllEqual(0.0, m1.result())
    self.assertAllEqual(2.0, m2.result())

  def testNamesWithSpaces(self):
    m1 = metrics.Mean("has space")
    m1(0)
    self.assertEqual(m1.name, "has space")
    self.assertEqual(m1.numer.name, "has_space/numer:0")

  def testGraphWithPlaceholder(self):
    with context.graph_mode(), self.test_session() as sess:
      m = metrics.Mean()
      p = array_ops.placeholder(dtypes.float32)
      accumulate = m(p)
      init_op = m.init_variables()
      init_op.run()
      sess.run(accumulate, feed_dict={p: [1, 10, 100]})
      sess.run(accumulate, feed_dict={p: 1000})
      sess.run(accumulate, feed_dict={p: [10000, 100000]})
      self.assertAllEqual(m.result().eval(), 111111.0/6)
      # Second init resets all the variables.
      init_op.run()
      sess.run(accumulate, feed_dict={p: 7})
      self.assertAllEqual(m.result().eval(), 7)

  @test_util.run_in_graph_and_eager_modes()
  def testGraphAndEagerTensor(self):
    m = metrics.Mean()
    inputs = ops.convert_to_tensor([1.0, 2.0])
    accumulate = m(inputs)
    result = m.result()
    self.evaluate(m.init_variables())
    self.evaluate(accumulate)
    self.assertEqual(self.evaluate(result), 1.5)
    # Second init resets all the variables.
    self.evaluate(m.init_variables())
    inputs = ops.convert_to_tensor([2.0, 3.0])
    self.evaluate(m(inputs))
    value = m.value()
    self.assertEqual(self.evaluate(value), 2.5)

  def testTwoMeansGraph(self):
    # Verify two metrics with the same name in the same graph raises a
    # ValueError.
    with context.graph_mode():
      m1 = metrics.Mean()
      m1(0)
      with self.assertRaises(ValueError):
        m2 = metrics.Mean()
        m2(2)

  def testBuildMean(self):
    # Verify that calling build() on Mean and then calling it won't recreate
    # variables.
    m = metrics.Mean()
    m.build()
    old_numer = m.numer
    m(0.0)
    self.assertTrue(old_numer is m.numer)

  def testMetricsChain(self):
    with context.graph_mode(), self.test_session():
      m1 = metrics.Mean()
      m2 = metrics.Mean(name="m2")
      update_m2 = m2(3.0)
      update_m2_2 = m2(m1(1.0))
      m1.init_variables().run()
      m2.init_variables().run()
      update_m2.eval()
      update_m2_2.eval()
      self.assertAllEqual(m2.result().eval(), 2.0)
      self.assertAllEqual(m1.result().eval(), 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    mean = metrics.Mean()
    checkpoint = checkpointable_utils.Checkpoint(mean=mean)
    mean.build()
    mean._built = True
    self.evaluate(mean.init_variables())
    self.evaluate(mean(100.))
    self.evaluate(mean(200.))
    save_path = checkpoint.save(checkpoint_prefix)
    self.evaluate(mean(1000.))
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.evaluate(mean(300.))
    self.assertAllEqual(200., self.evaluate(mean.value()))

    restore_mean = metrics.Mean()
    restore_checkpoint = checkpointable_utils.Checkpoint(mean=restore_mean)
    status = restore_checkpoint.restore(save_path)
    restore_update = restore_mean(300.)
    status.assert_consumed().run_restore_ops()
    self.evaluate(restore_update)
    self.assertAllEqual(200., self.evaluate(restore_mean.value()))
    self.assertEqual(3, self.evaluate(restore_mean.denom))

if __name__ == "__main__":
  test.main()
