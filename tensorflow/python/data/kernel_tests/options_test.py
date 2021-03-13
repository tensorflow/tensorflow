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
"""Tests for `tf.data.Options`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
import sys

from absl.testing import parameterized

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import stats_options
from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class OptionsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsDefault(self):
    ds = dataset_ops.Dataset.range(0)
    self.assertEqual(dataset_ops.Options(), ds.options())

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsOnce(self):
    options = dataset_ops.Options()
    ds = dataset_ops.Dataset.range(0).with_options(options).cache()
    self.assertEqual(options, ds.options())

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceSame(self):
    options = dataset_ops.Options()
    options.experimental_optimization.autotune = True
    ds = dataset_ops.Dataset.range(0).with_options(options).with_options(
        options)
    self.assertEqual(options, ds.options())

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceDifferentOptions(self):
    options1 = dataset_ops.Options()
    options1.experimental_optimization.autotune = True
    options2 = dataset_ops.Options()
    options2.experimental_deterministic = False
    ds = dataset_ops.Dataset.range(0)
    ds = ds.with_options(options1)
    ds = ds.with_options(options2)
    self.assertTrue(ds.options().experimental_optimization.autotune)
    # Explicitly check that flag is False since assertFalse allows None
    self.assertIs(ds.options().experimental_deterministic, False)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsTwiceSameOption(self):
    if sys.version_info >= (3, 8) and platform.system() == "Windows":
      # TODO(b/165013260): Fix this
      self.skipTest("Test is currently broken on Windows with Python 3.8")
    options1 = dataset_ops.Options()
    options1.experimental_optimization.autotune = False
    options2 = dataset_ops.Options()
    options2.experimental_optimization.autotune = True
    ds = dataset_ops.Dataset.range(0)
    ds = ds.with_options(options1)
    ds = ds.with_options(options2)
    self.assertTrue(ds.options().experimental_optimization.autotune)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsMergeOptionsFromMultipleInputs(self):
    options1 = dataset_ops.Options()
    options1.experimental_optimization.autotune = True
    options2 = dataset_ops.Options()
    options2.experimental_deterministic = True
    ds1 = dataset_ops.Dataset.range(0).with_options(options1)
    ds2 = dataset_ops.Dataset.range(0).with_options(options2)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    self.assertTrue(ds.options().experimental_optimization.autotune)
    self.assertTrue(ds.options().experimental_deterministic)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsHaveDefaults(self):
    options1 = dataset_ops.Options()
    options2 = dataset_ops.Options()
    self.assertIsNot(options1.experimental_optimization,
                     options2.experimental_optimization)
    self.assertIsNot(options1.experimental_stats,
                     options2.experimental_stats)
    self.assertIsNot(options1.experimental_threading,
                     options2.experimental_threading)
    self.assertEqual(options1.experimental_optimization,
                     optimization_options.OptimizationOptions())
    self.assertEqual(options1.experimental_stats, stats_options.StatsOptions())
    self.assertEqual(options1.experimental_threading,
                     threading_options.ThreadingOptions())

  @combinations.generate(test_base.default_test_combinations())
  def testMutatingOptionsRaiseValueError(self):
    ds = dataset_ops.Dataset.range(0)
    options1 = dataset_ops.Options()
    options1.experimental_slack = True
    options2 = dataset_ops.Options()
    options2.experimental_optimization.autotune = True
    ds = ds.with_options(options1)
    ds = ds.map(lambda x: 2 * x)
    ds = ds.with_options(options2)
    with self.assertRaises(ValueError):
      dataset_options = ds.options()
      dataset_options.experimental_deterministic = True

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedDataset(self):
    ds = dataset_ops.Dataset.from_tensors(0)
    result = ds

    for _ in range(999):
      result = result.concatenate(ds)
    self.assertDatasetProduces(result, [0]*1000)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsProtoRoundTrip(self):
    options = dataset_ops.Options()
    options.experimental_deterministic = True
    options.experimental_external_state_policy = (
        distribute_options.ExternalStatePolicy.FAIL)
    options.experimental_distribute.auto_shard_policy = (
        distribute_options.AutoShardPolicy.DATA)
    options.experimental_distribute.num_devices = 1000
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 10
    options.experimental_optimization.autotune_ram_budget = 20
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.filter_with_random_uniform_fusion = True
    options.experimental_optimization.hoist_random_uniform = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_vectorization.use_choose_fastest = (
        True)
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.reorder_data_discarding_ops = True
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_slack = True
    options.experimental_threading.max_intra_op_parallelism = 30
    options.experimental_threading.private_threadpool_size = 40
    pb = options._to_proto()
    result = dataset_ops.Options()
    result._from_proto(pb)
    self.assertEqual(options, result)

  @combinations.generate(test_base.default_test_combinations())
  def testOptionsProtoDefaultValuesRoundTrip(self):
    options = dataset_ops.Options()
    pb = options._to_proto()
    result = dataset_ops.Options()
    result._from_proto(pb)
    self.assertEqual(options, result)

  @combinations.generate(test_base.default_test_combinations())
  def testProtoOptionsDefaultValuesRoundTrip(self):
    pb = dataset_options_pb2.Options()
    options = dataset_ops.Options()
    options._from_proto(pb)
    result = options._to_proto()
    expected_pb = dataset_options_pb2.Options()
    expected_pb.distribute_options.CopyFrom(
        dataset_options_pb2.DistributeOptions())
    expected_pb.optimization_options.CopyFrom(
        dataset_options_pb2.OptimizationOptions())
    expected_pb.optimization_options.map_vectorization.CopyFrom(
        dataset_options_pb2.MapVectorization())
    expected_pb.threading_options.CopyFrom(
        dataset_options_pb2.ThreadingOptions())
    self.assertProtoEquals(expected_pb, result)


if __name__ == "__main__":
  test.main()
