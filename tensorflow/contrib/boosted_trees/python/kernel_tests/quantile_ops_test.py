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

import math
import os
import tempfile

import numpy as np

from tensorflow.contrib.boosted_trees.proto.quantiles_pb2 import QuantileConfig
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest
from tensorflow.python.training import saver


class QuantileBucketsOpTest(test_util.TensorFlowTestCase):

  def _gen_config(self, eps, num_quantiles):
    config = QuantileConfig()
    config.eps = eps
    config.num_quantiles = num_quantiles
    return config.SerializeToString()

  def testBasicQuantileBuckets(self):
    """Sets up the quantile summary op test as follows.

    Create a batch of 6 examples having a dense and sparse features. SparseM is
    a sparse multi-dimensional (multivalent) feature.
    The data looks like this
    | Instance | instance weights | Dense 0  | Sparse 0 | SparseM
    | 0        |     10           |   1      |          |   |   |
    | 1        |     1            |   2      |    2     | 2 |   |
    | 2        |     1            |   3      |    3     | 3 |   |
    | 3        |     1            |   4      |    4     |   | 4 |
    | 4        |     1            |   4      |    5     |   | 5 |
    | 5        |     1            |   5      |    6     |   | 6 |
    """

    dense_float_tensor_0 = constant_op.constant(
        [1, 2, 3, 4, 4, 5], dtype=dtypes.float32)
    sparse_indices_0 = constant_op.constant(
        [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=dtypes.int64)
    sparse_values_0 = constant_op.constant(
        [2, 3, 4, 5, 6], dtype=dtypes.float32)
    sparse_shape_0 = constant_op.constant([6, 1], dtype=dtypes.int64)
    # Multi-dimensional feature that should have the same quantiles as Sparse 0.
    sparse_indices_m = constant_op.constant(
        [[1, 1], [2, 0], [3, 1], [4, 1], [5, 1]], dtype=dtypes.int64)
    sparse_values_m = constant_op.constant(
        [2, 3, 4, 5, 6], dtype=dtypes.float32)
    sparse_shape_m = constant_op.constant([6, 2], dtype=dtypes.int64)

    example_weights = constant_op.constant(
        [10, 1, 1, 1, 1, 1], dtype=dtypes.float32)

    with self.cached_session():
      config = self._gen_config(0.33, 3)
      dense_buckets, sparse_buckets = quantile_ops.quantile_buckets(
          [dense_float_tensor_0], [sparse_indices_0, sparse_indices_m],
          [sparse_values_0, sparse_values_m], [sparse_shape_0, sparse_shape_m],
          example_weights=example_weights,
          dense_config=[config],
          sparse_config=[config, config])

      self.assertAllEqual([1, 3, 5], dense_buckets[0].eval())
      self.assertAllEqual([2, 4, 6.], sparse_buckets[0].eval())
      # Multidimensional sparse.
      self.assertAllEqual([2, 4, 6.], sparse_buckets[1].eval())

  def testStreamingQuantileBucketsWithVaryingBatch(self):
    """Sets up the quantile summary op test as follows.

    Creates batches examples with different number of inputs in each batch.
    The input values are dense in the range [1 ... N]
    The data looks like this:
    | Batch | Start | InputList
    |   1   |   1   |  [1]
    |   2   |   2   |  [2, 3]
    |   3   |   4   |  [4, 5, 6]
    |   4   |   7   |  [7, 8, 9, 10]
    |   5   |  11   |  [11, 12, 13, 14, 15]
    |   6   |  16   |  [16, 17, 18, 19, 20, 21]
    """

    num_quantiles = 3
    with self.cached_session() as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=num_quantiles,
          epsilon=0.001, name="q1")
      resources.initialize_resources(resources.shared_resources()).run()
    input_column = array_ops.placeholder(dtypes.float32)
    weights = array_ops.placeholder(dtypes.float32)
    update = accumulator.add_summary(
        stamp_token=0,
        column=input_column,
        example_weights=weights)

    with self.cached_session() as sess:
      for i in range(1, 23):
        # start = 1, 2, 4, 7, 11, 16 ... (see comment above)
        start = int((i * (i-1) / 2) + 1)
        sess.run(update,
                 {input_column: range(start, start+i),
                  weights: [1] * i})

    with self.cached_session() as sess:
      sess.run(accumulator.flush(stamp_token=0, next_stamp_token=1))
      are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run(
          [buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertEqual(num_quantiles + 1, len(buckets))
      self.assertAllEqual([1, 86., 170., 253.], buckets)

  def testStreamingQuantileBucketsLowPrecisionInput(self):
    """Tests inputs that simulate low precision float16 values."""

    num_quantiles = 3
    # set generate_quantiles to True since the test will generate fewer
    # boundaries otherwise.
    with self.cached_session() as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=num_quantiles,
          epsilon=0.001, name="q1", generate_quantiles=True)
      resources.initialize_resources(resources.shared_resources()).run()
    input_column = array_ops.placeholder(dtypes.float32)
    weights = array_ops.placeholder(dtypes.float32)
    update = accumulator.add_summary(
        stamp_token=0,
        column=input_column,
        example_weights=weights)

    with self.cached_session() as sess:
      # This input is generated by integer in the range [2030, 2060]
      # but represented by with float16 precision. Integers <= 2048 are
      # exactly represented, whereas  numbers > 2048 are rounded; and hence
      # numbers > 2048 are repeated. For precision loss / rounding, see:
      # https://en.wikipedia.org/wiki/Half-precision_floating-point_format.
      #
      # The intent of the test is not handling of float16 values, but to
      # validate the number of buckets is returned, in cases where  the input
      # may contain repeated values.
      inputs = [
          2030.0, 2031.0, 2032.0, 2033.0, 2034.0, 2035.0, 2036.0, 2037.0,
          2038.0, 2039.0, 2040.0, 2041.0, 2042.0, 2043.0, 2044.0, 2045.0,
          2046.0, 2047.0, 2048.0, 2048.0, 2050.0, 2052.0, 2052.0, 2052.0,
          2054.0, 2056.0, 2056.0, 2056.0, 2058.0, 2060.0
      ]
      sess.run(update,
               {input_column: inputs,
                weights: [1] * len(inputs)})

    with self.cached_session() as sess:
      sess.run(accumulator.flush(stamp_token=0, next_stamp_token=1))
      are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run(
          [buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertEqual(num_quantiles + 1, len(buckets))
      self.assertAllEqual([2030, 2040, 2050, 2060], buckets)

  def _testStreamingQuantileBucketsHelper(
      self, inputs, num_quantiles=3, expected_buckets=None):
    """Helper to test quantile buckets on different inputs."""

    # set generate_quantiles to True since the test will generate fewer
    # boundaries otherwise.
    with self.cached_session() as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=num_quantiles,
          epsilon=0.001, name="q1", generate_quantiles=True)
      resources.initialize_resources(resources.shared_resources()).run()
    input_column = array_ops.placeholder(dtypes.float32)
    weights = array_ops.placeholder(dtypes.float32)
    update = accumulator.add_summary(
        stamp_token=0,
        column=input_column,
        example_weights=weights)

    with self.cached_session() as sess:
      sess.run(update,
               {input_column: inputs,
                weights: [1] * len(inputs)})

    with self.cached_session() as sess:
      sess.run(accumulator.flush(stamp_token=0, next_stamp_token=1))
      are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run(
          [buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      # By default, use 3 quantiles, 4 boundaries for simplicity.
      self.assertEqual(num_quantiles + 1, len(buckets))
      if expected_buckets:
        self.assertAllEqual(buckets, expected_buckets)

  def testStreamingQuantileBucketsRepeatedSingleValue(self):
    inputs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    self._testStreamingQuantileBucketsHelper(inputs)

  def testStreamingQ2antileBucketsRepeatedTwoValues(self):
    inputs = [1, 1, 1, 2, 2, 2, 2, 2, 1, 1]
    self._testStreamingQuantileBucketsHelper(inputs)

  def testStreamingQ2antileBucketsRepeatedTwoValuesUnbalanced(self):
    inputs = [7, 7, 7, 2, 7, 7, 2, 2, 7, 7]
    self._testStreamingQuantileBucketsHelper(inputs)

  def testStreamingQuantileBucketsFewerInputstThanBuckets(self):
    inputs = [5]
    self._testStreamingQuantileBucketsHelper(inputs)

  def testStreamingQuantileBucketsEqualDistributionInSequence(self):
    # Input pattern is of the form [1, 1, 1, 2, 2, 2, 3, 3, 3, ...]
    ones = 100 * [1]
    inputs = []
    for i in range(1, 101):
      inputs += [i * k for k in ones]
    # Expect 100 equally spaced buckets.
    expected_buckets = range(1, 101)
    self._testStreamingQuantileBucketsHelper(
        inputs, num_quantiles=99, expected_buckets=expected_buckets)

  def testStreamingQuantileBucketsEqualDistributionInterleaved(self):
    # Input pattern is of the form [1, 2, 3, 1, 2, 3, 1, 2, 3, ...]
    sequence = range(1, 101)
    inputs = []
    for _ in range(1, 101):
      inputs += sequence
    # Expect 100 equally spaced buckets.
    expected_buckets = range(1, 101)
    self._testStreamingQuantileBucketsHelper(
        inputs, num_quantiles=99, expected_buckets=expected_buckets)

  def testStreamingQuantileBuckets(self):
    """Sets up the quantile summary op test as follows.

    100 batches of data is added to the accumulator. The batches are in form:
    [0 1 .. 99]
    [100 101 .. 200]
    ...
    [9900 9901 .. 9999]
    All the batches have 1 for all the example weights.
    """
    with self.cached_session() as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.01, name="q1")
      resources.initialize_resources(resources.shared_resources()).run()
    weight_placeholder = array_ops.placeholder(dtypes.float32)
    dense_placeholder = array_ops.placeholder(dtypes.float32)
    update = accumulator.add_summary(
        stamp_token=0,
        column=dense_placeholder,
        example_weights=weight_placeholder)
    with self.cached_session() as sess:
      for i in range(100):
        dense_float = np.linspace(
            i * 100, (i + 1) * 100 - 1, num=100).reshape(-1, 1)
        sess.run(update, {
            dense_placeholder: dense_float,
            weight_placeholder: np.ones(shape=(100, 1), dtype=np.float32)
        })

    with self.cached_session() as sess:
      sess.run(accumulator.flush(stamp_token=0, next_stamp_token=1))
      are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run([buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertAllEqual([0, 3335., 6671., 9999.], buckets)

  def testStreamingQuantileBucketsTwoLevel(self):
    """Sets up the quantile summary op test as follows.

    100 batches of data is added to the accumulator. The batches are in form:
    [0 1 .. 99]
    [100 101 .. 200]
    ...
    [9900 9901 .. 9999]
    All the batches have 1 for all the example weights.
    """
    with self.cached_session() as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.01, name="q1")
      accumulator_2 = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.01, name="q2")
      resources.initialize_resources(resources.shared_resources()).run()
    weight_placeholder = array_ops.placeholder(dtypes.float32)
    dense_placeholder = array_ops.placeholder(dtypes.float32)
    update = accumulator.add_summary(
        stamp_token=0,
        column=dense_placeholder,
        example_weights=weight_placeholder)
    with self.cached_session() as sess:
      for i in range(100):
        dense_float = np.linspace(
            i * 100, (i + 1) * 100 - 1, num=100).reshape(-1, 1)
        sess.run(update, {
            dense_placeholder: dense_float,
            weight_placeholder: np.ones(shape=(100, 1), dtype=np.float32)
        })

    with self.cached_session() as sess:
      summary = sess.run(
          accumulator.flush_summary(stamp_token=0, next_stamp_token=1))
      sess.run(
          accumulator_2.add_prebuilt_summary(
              stamp_token=0, summary=constant_op.constant(summary)))
      sess.run(accumulator_2.flush(stamp_token=0, next_stamp_token=1))
      are_ready_flush, buckets = (accumulator_2.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run([buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertAllEqual([0, 3337., 6677., 9999.], buckets)

  def testSaveRestoreBeforeFlush(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.33, name="q0")

      save = saver.Saver()
      resources.initialize_resources(resources.shared_resources()).run()

      sparse_indices_0 = constant_op.constant(
          [[1, 0], [2, 1], [3, 0], [4, 2], [5, 0]], dtype=dtypes.int64)
      sparse_values_0 = constant_op.constant(
          [2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtypes.float32)
      sparse_shape_0 = constant_op.constant([6, 3], dtype=dtypes.int64)
      example_weights = constant_op.constant(
          [10, 1, 1, 1, 1, 1], dtype=dtypes.float32, shape=[6, 1])
      update = accumulator.add_summary(
          stamp_token=0,
          column=sparse_tensor.SparseTensor(sparse_indices_0, sparse_values_0,
                                            sparse_shape_0),
          example_weights=example_weights)
      update.run()
      save.save(sess, save_path)
      reset = accumulator.flush(stamp_token=0, next_stamp_token=1)
      with ops.control_dependencies([reset]):
        are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run([buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertAllEqual([2, 4, 6.], buckets)

    with self.session(graph=ops.Graph()) as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.33, name="q0")
      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      are_ready_noflush = accumulator.get_buckets(stamp_token=0)[0]
      with ops.control_dependencies([are_ready_noflush]):
        reset = accumulator.flush(stamp_token=0, next_stamp_token=1)

      with ops.control_dependencies([reset]):
        are_ready_flush, buckets = accumulator.get_buckets(stamp_token=1)
      buckets, are_ready_flush, are_ready_noflush = (sess.run(
          [buckets, are_ready_flush, are_ready_noflush]))
      self.assertFalse(are_ready_noflush)
      self.assertTrue(are_ready_flush)
      self.assertAllEqual([2, 4, 6.], buckets)

  def testSaveRestoreAfterFlush(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.33, name="q0")

      save = saver.Saver()
      resources.initialize_resources(resources.shared_resources()).run()

      example_weights = constant_op.constant(
          [10, 1, 1, 1, 1, 1], dtype=dtypes.float32, shape=[6, 1])
      dense_float_tensor_0 = constant_op.constant(
          [1, 2, 3, 4, 4, 5], dtype=dtypes.float32, shape=[6, 1])
      update = accumulator.add_summary(
          stamp_token=0,
          column=dense_float_tensor_0,
          example_weights=example_weights)
      update.run()
      reset = accumulator.flush(stamp_token=0, next_stamp_token=1)
      with ops.control_dependencies([reset]):
        are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run([buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertAllEqual([1, 3, 5], buckets)
      save.save(sess, save_path)

    with self.session(graph=ops.Graph()) as sess:
      accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token=0, num_quantiles=3, epsilon=0.33, name="q0")
      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      are_ready_flush, buckets = (accumulator.get_buckets(stamp_token=1))
      buckets, are_ready_flush = (sess.run([buckets, are_ready_flush]))
      self.assertEqual(True, are_ready_flush)
      self.assertAllEqual([1, 3, 5], buckets)

  def testFixedUniform(self):
    """Sets up the quantile summary op test as follows.

    Creates array dividing range [0, 1] to 1<<16 elements equally spaced
    with weight of 1.0.
    """
    dense_float_tensor_0 = constant_op.constant(
        [(1.0 * i) / math.pow(2.0, 16)
         for i in range(0, int(math.pow(2, 16)) + 1)])
    example_weights = constant_op.constant(
        [1] * (int(math.pow(2, 16)) + 1), dtype=dtypes.float32)
    config = self._gen_config(0.1, 10)

    with self.cached_session():
      dense_buckets, _ = quantile_ops.quantile_buckets(
          [dense_float_tensor_0], [], [], [],
          example_weights=example_weights,
          dense_config=[config],
          sparse_config=[])
      self.assertAllClose(
          [0] + [(i + 1.0) / 10 for i in range(0, 10)],
          dense_buckets[0].eval(),
          atol=0.1)

  def testFixedNonUniform(self):
    """Sets up the quantile summary op test as follows.

    Creates array dividing range [0, 1] to 1<<16 elements equally spaced
    with weight same as the value.
    """
    dense_float_tensor_0 = constant_op.constant(
        [(1.0 * i) / math.pow(2.0, 16)
         for i in range(0, int(math.pow(2, 16)) + 1)])
    example_weights = constant_op.constant(
        [(1.0 * i) / math.pow(2.0, 16)
         for i in range(0, int(math.pow(2, 16)) + 1)])

    config = self._gen_config(0.1, 10)

    with self.cached_session():
      dense_buckets, _ = quantile_ops.quantile_buckets(
          [dense_float_tensor_0], [], [], [],
          example_weights=example_weights,
          dense_config=[config],
          sparse_config=[])
      self.assertAllClose(
          [0] + [math.sqrt((i + 1.0) / 10) for i in range(0, 10)],
          dense_buckets[0].eval(),
          atol=0.1)


class QuantilesOpTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Sets up the quantile op tests.

    Create a batch of 4 examples having 2 dense and 4 sparse features.
    Fourth sparse feature is multivalent (3 dimensional)
    The data looks like this
    | Instance | Dense 0 | Dense 1 | Sparse 0 | Sparse 1 |Sparse 2| SparseM
    | 0        |   -0.1  |  -1     |   -2     |   0.1    |        |_ ,1,_
    | 1        |    0.4  |  -15    |   5.5    |          |  2     |2 ,_,_
    | 2        |    3.2  |  18     |   16     |   3      |        |__,_,_
    | 3        |    190  |  1000   |   17.5   |  -3      |  4     |1 ,8,1
    Quantiles are:
    Dense 0: (-inf,0.4], (0.4,5], (5, 190]
    Dense 1: (-inf, -9], (-9,15], (15, 1000)
    Sparse 0: (-inf, 5], (5,16], (16, 100]
    Sparse 1: (-inf, 2], (2, 5]
    Sparse 2: (-inf, 100]
    SparseM: (-inf, 1], (1,2], (2,1000]
    """
    super(QuantilesOpTest, self).setUp()
    self._dense_float_tensor_0 = constant_op.constant(
        [[-0.1], [0.4], [3.2], [190]], dtype=dtypes.float32)
    self._dense_float_tensor_1 = constant_op.constant(
        [[-1], [-15], [18], [1000]], dtype=dtypes.float32)
    # Sparse feature 0
    self._sparse_indices_0 = constant_op.constant(
        [[0, 0], [1, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
    self._sparse_values_0 = constant_op.constant([-2, 5.5, 16, 17.5])
    self._sparse_shape_0 = constant_op.constant([4, 1])
    # Sprase feature 1
    self._sparse_indices_1 = constant_op.constant(
        [[0, 0], [2, 0], [3, 0]], dtype=dtypes.int64)
    self._sparse_values_1 = constant_op.constant([0.1, 3, -3])
    self._sparse_shape_1 = constant_op.constant([4, 1])
    # Sprase feature 2
    self._sparse_indices_2 = constant_op.constant(
        [[1, 0], [3, 0]], dtype=dtypes.int64)
    self._sparse_values_2 = constant_op.constant([2, 4], dtype=dtypes.float32)
    self._sparse_shape_2 = constant_op.constant([4, 1])
    # Sprase feature M
    self._sparse_indices_m = constant_op.constant(
        [[0, 1], [1, 0], [3, 0], [3, 1], [3, 2]], dtype=dtypes.int64)
    self._sparse_values_m = constant_op.constant(
        [1, 2, 1, 8, 1], dtype=dtypes.float32)
    self._sparse_shape_m = constant_op.constant([4, 1])
    # Quantiles
    self._dense_thresholds_0 = [0.4, 5, 190]
    self._dense_thresholds_1 = [-9, 15, 1000]

    self._sparse_thresholds_0 = [5, 16, 100]
    self._sparse_thresholds_1 = [2, 5]
    self._sparse_thresholds_2 = [100]
    self._sparse_thresholds_m = [1, 2, 1000]

  def testDenseFeaturesOnly(self):
    with self.cached_session():
      dense_quantiles, _ = quantile_ops.quantiles(
          [self._dense_float_tensor_0, self._dense_float_tensor_1], [],
          [self._dense_thresholds_0, self._dense_thresholds_1], [], [])

      # Dense feature 0
      self.assertAllEqual([[0, 0], [0, 0], [1, 0], [2, 0]],
                          dense_quantiles[0].eval())
      # Dense feature 1
      self.assertAllEqual([[1, 0], [0, 0], [2, 0], [2, 0]],
                          dense_quantiles[1].eval())

  def testSparseFeaturesOnly(self):
    with self.cached_session():
      _, sparse_quantiles = quantile_ops.quantiles([], [
          self._sparse_values_0, self._sparse_values_1, self._sparse_values_2,
          self._sparse_values_m
      ], [], [
          self._sparse_thresholds_0, self._sparse_thresholds_1,
          self._sparse_thresholds_2, self._sparse_thresholds_m
      ], [
          self._sparse_indices_0, self._sparse_indices_1,
          self._sparse_indices_2, self._sparse_indices_m
      ])

      self.assertAllEqual(4, len(sparse_quantiles))
      # Sparse feature 0
      self.assertAllEqual([[0, 0], [1, 0], [1, 0], [2, 0]],
                          sparse_quantiles[0].eval())
      # Sparse feature 1
      self.assertAllEqual([[0, 0], [1, 0], [0, 0]], sparse_quantiles[1].eval())
      # Sparse feature 2
      self.assertAllEqual([[0, 0], [0, 0]], sparse_quantiles[2].eval())
      # Multidimensional feature.
      self.assertAllEqual([[0, 1], [1, 0], [0, 0], [2, 1], [0, 2]],
                          sparse_quantiles[3].eval())

  def testDenseAndSparseFeatures(self):
    with self.cached_session():
      dense_quantiles, sparse_quantiles = quantile_ops.quantiles(
          [self._dense_float_tensor_0, self._dense_float_tensor_1], [
              self._sparse_values_0, self._sparse_values_1,
              self._sparse_values_2, self._sparse_values_m
          ], [self._dense_thresholds_0, self._dense_thresholds_1], [
              self._sparse_thresholds_0, self._sparse_thresholds_1,
              self._sparse_thresholds_2, self._sparse_thresholds_m
          ], [
              self._sparse_indices_0, self._sparse_indices_1,
              self._sparse_indices_2, self._sparse_indices_m
          ])

      # Dense feature 0
      self.assertAllEqual([[0, 0], [0, 0], [1, 0], [2, 0]],
                          dense_quantiles[0].eval())
      # Dense feature 1
      self.assertAllEqual([[1, 0], [0, 0], [2, 0], [2, 0]],
                          dense_quantiles[1].eval())
      # Sparse feature 0
      self.assertAllEqual([[0, 0], [1, 0], [1, 0], [2, 0]],
                          sparse_quantiles[0].eval())
      # Sparse feature 1
      self.assertAllEqual([[0, 0], [1, 0], [0, 0]], sparse_quantiles[1].eval())
      # Sparse feature 2
      self.assertAllEqual([[0, 0], [0, 0]], sparse_quantiles[2].eval())
      # Multidimensional feature.
      self.assertAllEqual([[0, 1], [1, 0], [0, 0], [2, 1], [0, 2]],
                          sparse_quantiles[3].eval())

  def testBucketizeWithInputBoundaries(self):
    with self.cached_session():
      buckets = quantile_ops.bucketize_with_input_boundaries(
          input=[1, 2, 3, 4, 5],
          boundaries=[3])
      self.assertAllEqual([0, 0, 1, 1, 1], buckets.eval())

  def testBucketizeWithInputBoundaries2(self):
    with self.cached_session():
      boundaries = constant_op.constant([3], dtype=dtypes.float32)
      buckets = quantile_ops.bucketize_with_input_boundaries(
          input=[1, 2, 3, 4, 5],
          boundaries=boundaries)
      self.assertAllEqual([0, 0, 1, 1, 1], buckets.eval())

  def testBucketizeWithInputBoundaries3(self):
    with self.cached_session():
      b = array_ops.placeholder(dtypes.float32)
      buckets = quantile_ops.bucketize_with_input_boundaries(
          input=[1, 2, 3, 4, 5],
          boundaries=b)
      self.assertAllEqual([0, 1, 1, 2, 2],
                          buckets.eval(feed_dict={b: [2, 4]}))

if __name__ == "__main__":
  googletest.main()
