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
"""Test for checking quantile related ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as resource_initialized
from tensorflow.python.platform import googletest
from tensorflow.python.training import saver


class QuantileOpsTest(test_util.TensorFlowTestCase):

  def create_resource(self, name, eps, max_elements, num_streams=1):
    quantile_accumulator_handle = resource_handle_op(
        container="", shared_name=name, name=name)
    create_op = boosted_trees_ops.create_quantile_stream_resource(
        quantile_accumulator_handle,
        epsilon=eps,
        max_elements=max_elements,
        num_streams=num_streams)
    is_initialized_op = resource_initialized(quantile_accumulator_handle)
    resources.register_resource(quantile_accumulator_handle, create_op,
                                is_initialized_op)
    return quantile_accumulator_handle

  def setUp(self):
    """Sets up the quantile ops test as follows.

    Create a batch of 6 examples having 2 features
    The data looks like this
    | Instance | instance weights | Feature 0 | Feature 1
    | 0        |     10           |   1.2     |   2.3
    | 1        |     1            |   12.1    |   1.2
    | 2        |     1            |   0.3     |   1.1
    | 3        |     1            |   0.5     |   2.6
    | 4        |     1            |   0.6     |   3.2
    | 5        |     1            |   2.2     |   0.8
    """

    self._feature_0 = constant_op.constant([1.2, 12.1, 0.3, 0.5, 0.6, 2.2],
                                           dtype=dtypes.float32)
    self._feature_1 = constant_op.constant([2.3, 1.2, 1.1, 2.6, 3.2, 0.8],
                                           dtype=dtypes.float32)
    self._feature_0_boundaries = np.array([0.3, 0.6, 1.2, 12.1])
    self._feature_1_boundaries = np.array([0.8, 1.2, 2.3, 3.2])
    self._feature_0_quantiles = constant_op.constant([2, 3, 0, 1, 1, 3],
                                                     dtype=dtypes.int32)
    self._feature_1_quantiles = constant_op.constant([2, 1, 1, 3, 3, 0],
                                                     dtype=dtypes.int32)

    self._example_weights = constant_op.constant(
        [10, 1, 1, 1, 1, 1], dtype=dtypes.float32)

    self.eps = 0.01
    self.max_elements = 1 << 16
    self.num_quantiles = constant_op.constant(3, dtype=dtypes.int64)

  def testBasicQuantileBucketsSingleResource(self):
    with self.cached_session() as sess:
      quantile_accumulator_handle = self.create_resource("floats", self.eps,
                                                         self.max_elements, 2)
      resources.initialize_resources(resources.shared_resources()).run()
      summaries = boosted_trees_ops.make_quantile_summaries(
          [self._feature_0, self._feature_1], self._example_weights,
          epsilon=self.eps)
      summary_op = boosted_trees_ops.quantile_add_summaries(
          quantile_accumulator_handle, summaries)
      flush_op = boosted_trees_ops.quantile_flush(
          quantile_accumulator_handle, self.num_quantiles)
      buckets = boosted_trees_ops.get_bucket_boundaries(
          quantile_accumulator_handle, num_features=2)
      quantiles = boosted_trees_ops.boosted_trees_bucketize(
          [self._feature_0, self._feature_1], buckets)
      self.evaluate(summary_op)
      self.evaluate(flush_op)
      self.assertAllClose(self._feature_0_boundaries, buckets[0].eval())
      self.assertAllClose(self._feature_1_boundaries, buckets[1].eval())

      self.assertAllClose(self._feature_0_quantiles, quantiles[0].eval())
      self.assertAllClose(self._feature_1_quantiles, quantiles[1].eval())

  def testBasicQuantileBucketsMultipleResources(self):
    with self.cached_session() as sess:
      quantile_accumulator_handle_0 = self.create_resource("float_0", self.eps,
                                                           self.max_elements)
      quantile_accumulator_handle_1 = self.create_resource("float_1", self.eps,
                                                           self.max_elements)
      resources.initialize_resources(resources.shared_resources()).run()
      summaries = boosted_trees_ops.make_quantile_summaries(
          [self._feature_0, self._feature_1], self._example_weights,
          epsilon=self.eps)
      summary_op_0 = boosted_trees_ops.quantile_add_summaries(
          quantile_accumulator_handle_0,
          [summaries[0]])
      summary_op_1 = boosted_trees_ops.quantile_add_summaries(
          quantile_accumulator_handle_1,
          [summaries[1]])
      flush_op_0 = boosted_trees_ops.quantile_flush(
          quantile_accumulator_handle_0, self.num_quantiles)
      flush_op_1 = boosted_trees_ops.quantile_flush(
          quantile_accumulator_handle_1, self.num_quantiles)
      bucket_0 = boosted_trees_ops.get_bucket_boundaries(
          quantile_accumulator_handle_0, num_features=1)
      bucket_1 = boosted_trees_ops.get_bucket_boundaries(
          quantile_accumulator_handle_1, num_features=1)
      quantiles = boosted_trees_ops.boosted_trees_bucketize(
          [self._feature_0, self._feature_1], bucket_0 + bucket_1)
      self.evaluate([summary_op_0, summary_op_1])
      self.evaluate([flush_op_0, flush_op_1])
      self.assertAllClose(self._feature_0_boundaries, bucket_0[0].eval())
      self.assertAllClose(self._feature_1_boundaries, bucket_1[0].eval())

      self.assertAllClose(self._feature_0_quantiles, quantiles[0].eval())
      self.assertAllClose(self._feature_1_quantiles, quantiles[1].eval())

  def testSaveRestoreAfterFlush(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session() as sess:
      accumulator = boosted_trees_ops.QuantileAccumulator(
          num_streams=2, num_quantiles=3, epsilon=self.eps, name="q0")

      save = saver.Saver()
      resources.initialize_resources(resources.shared_resources()).run()

      buckets = accumulator.get_bucket_boundaries()
      self.assertAllClose([], buckets[0].eval())
      self.assertAllClose([], buckets[1].eval())
      summaries = accumulator.add_summaries([self._feature_0, self._feature_1],
                                            self._example_weights)
      with ops.control_dependencies([summaries]):
        flush = accumulator.flush()
      self.evaluate(flush)
      self.assertAllClose(self._feature_0_boundaries, buckets[0].eval())
      self.assertAllClose(self._feature_1_boundaries, buckets[1].eval())
      save.save(sess, save_path)

    with self.test_session(graph=ops.Graph()) as sess:
      accumulator = boosted_trees_ops.QuantileAccumulator(
          num_streams=2, num_quantiles=3, epsilon=self.eps, name="q0")
      save = saver.Saver()
      save.restore(sess, save_path)
      buckets = accumulator.get_bucket_boundaries()
      self.assertAllClose(self._feature_0_boundaries, buckets[0].eval())
      self.assertAllClose(self._feature_1_boundaries, buckets[1].eval())

  def testSaveRestoreBeforeFlush(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session() as sess:
      accumulator = boosted_trees_ops.QuantileAccumulator(
          num_streams=2, num_quantiles=3, epsilon=self.eps, name="q0")

      save = saver.Saver()
      resources.initialize_resources(resources.shared_resources()).run()

      summaries = accumulator.add_summaries([self._feature_0, self._feature_1],
                                            self._example_weights)
      self.evaluate(summaries)
      buckets = accumulator.get_bucket_boundaries()
      self.assertAllClose([], buckets[0].eval())
      self.assertAllClose([], buckets[1].eval())
      save.save(sess, save_path)
      self.evaluate(accumulator.flush())
      self.assertAllClose(self._feature_0_boundaries, buckets[0].eval())
      self.assertAllClose(self._feature_1_boundaries, buckets[1].eval())

    with self.test_session(graph=ops.Graph()) as sess:
      accumulator = boosted_trees_ops.QuantileAccumulator(
          num_streams=2, num_quantiles=3, epsilon=self.eps, name="q0")
      save = saver.Saver()
      save.restore(sess, save_path)
      buckets = accumulator.get_bucket_boundaries()
      self.assertAllClose([], buckets[0].eval())
      self.assertAllClose([], buckets[1].eval())


if __name__ == "__main__":
  googletest.main()
