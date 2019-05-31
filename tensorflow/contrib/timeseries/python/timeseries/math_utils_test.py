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
"""Tests for math_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl


class MathUtilsTest(test.TestCase):

  def setUp(self):
    numpy.random.seed(10)

  def test_power_sums_tensor(self):
    transition = numpy.random.normal(size=[4, 4]).astype(numpy.float32)
    addition = numpy.random.normal(size=[4, 4]).astype(numpy.float32)
    array_size = 2
    result = []
    transition_power = numpy.identity(4)
    running_sum = numpy.zeros([4, 4], dtype=numpy.float32)
    for _ in range(array_size + 1):
      result.append(running_sum)
      current_contribution = numpy.dot(numpy.dot(transition_power, addition),
                                       transition_power.T)
      # pylint: disable=g-no-augmented-assignment
      # += has different semantics here; want to make a copy
      running_sum = running_sum + current_contribution
      # pylint: enable=g-no-augmented-assignment
      transition_power = numpy.dot(transition, transition_power)
    with self.cached_session():
      self.assertAllClose(result,
                          math_utils.power_sums_tensor(
                              array_size, transition, addition).eval())

  def test_matrix_to_powers(self):
    matrix = numpy.random.normal(size=[4, 4]).astype(numpy.float32)
    powers = numpy.random.randint(low=0, high=10, size=20)
    result = []
    for i in range(powers.shape[0]):
      result.append(numpy.linalg.matrix_power(matrix, powers[i]))
    with self.cached_session():
      self.assertAllClose(result,
                          math_utils.matrix_to_powers(matrix, powers).eval(),
                          rtol=1e-5,
                          atol=1e-5)

  def test_batch_matrix_pow(self):
    batch = numpy.random.normal(size=[15, 4, 4]).astype(numpy.float32)
    powers = numpy.random.randint(low=0, high=10, size=batch.shape[0])
    result = []
    for i in range(batch.shape[0]):
      result.append(numpy.linalg.matrix_power(batch[i], powers[i]))
    with self.cached_session():
      # TODO(allenl): Numerical errors seem to be creeping in. Maybe it can be
      # made slightly more stable?
      self.assertAllClose(result,
                          math_utils.batch_matrix_pow(batch, powers).eval(),
                          rtol=1e-5,
                          atol=1e-5)

  def test_batch_times_matrix(self):
    left = numpy.random.normal(size=[5, 3, 2]).astype(numpy.float32)
    left_transpose = numpy.transpose(left, [0, 2, 1])
    right = numpy.random.normal(size=[2, 3]).astype(numpy.float32)
    expected_result = numpy.dot(left, right)
    with self.cached_session():
      self.assertAllClose(expected_result,
                          math_utils.batch_times_matrix(
                              left, right).eval())
      self.assertAllClose(expected_result,
                          math_utils.batch_times_matrix(
                              left_transpose, right,
                              adj_x=True).eval())
      self.assertAllClose(expected_result,
                          math_utils.batch_times_matrix(
                              left, right.T,
                              adj_y=True).eval())
      self.assertAllClose(expected_result,
                          math_utils.batch_times_matrix(
                              left_transpose, right.T,
                              adj_x=True, adj_y=True).eval())

  def test_matrix_times_batch(self):
    left = numpy.random.normal(size=[5, 7]).astype(numpy.float32)
    right = numpy.random.normal(size=[3, 7, 9]).astype(numpy.float32)
    right_transpose = numpy.transpose(right, [0, 2, 1])
    expected_result = numpy.transpose(numpy.dot(right_transpose, left.T),
                                      [0, 2, 1])
    with self.cached_session():
      self.assertAllClose(expected_result,
                          math_utils.matrix_times_batch(
                              left, right).eval())
      self.assertAllClose(expected_result,
                          math_utils.matrix_times_batch(
                              left.T, right,
                              adj_x=True).eval())
      self.assertAllClose(expected_result,
                          math_utils.matrix_times_batch(
                              left, right_transpose,
                              adj_y=True).eval())
      self.assertAllClose(expected_result,
                          math_utils.matrix_times_batch(
                              left.T, right_transpose,
                              adj_x=True, adj_y=True).eval())

  def test_make_diagonal_undefined_shapes(self):
    with self.cached_session():
      completely_undefined = array_ops.placeholder(dtype=dtypes.float32)
      partly_undefined = array_ops.placeholder(
          shape=[None, None], dtype=dtypes.float32)
      blocked = math_utils.block_diagonal([completely_undefined,
                                           [[2.]],
                                           partly_undefined])
      self.assertEqual([None, None],
                       blocked.get_shape().as_list())
      self.assertAllEqual(
          [[1., 0., 0., 0.],
           [0., 2., 0., 0.],
           [0., 0., 3., 4.],
           [0., 0., 5., 6.]],
          blocked.eval(feed_dict={
              completely_undefined: [[1.]],
              partly_undefined: [[3., 4.],
                                 [5., 6.]]}))

  def test_make_diagonal_mostly_defined_shapes(self):
    with self.cached_session():
      mostly_defined = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      blocked = math_utils.block_diagonal([[[2.]],
                                           mostly_defined,
                                           [[7.]]])
      self.assertEqual([None, 4],
                       blocked.get_shape().as_list())
      self.assertAllEqual(
          [[2., 0., 0., 0.],
           [0., 3., 4., 0.],
           [0., 5., 6., 0.],
           [0., 0., 0., 7.]],
          blocked.eval(feed_dict={
              mostly_defined: [[3., 4.],
                               [5., 6.]]}))


class TestMakeToeplitzMatrix(test.TestCase):

  def test_make_toeplitz_matrix_1(self):
    inputs = numpy.array([[[1.]], [[2.]], [[3.]]])
    output_expected = numpy.array([[1., 2, 3], [2, 1, 2], [3, 2, 1]])
    self._test_make_toeplitz_matrix(inputs, output_expected)

  def test_make_toeplitz_matrix_2(self):
    inputs = numpy.array(
        [[[1, 2.], [3, 4]], [[5, 6], [7, 8]], [[8, 9], [10, 11]]])

    output_expected = numpy.array(
        [[1., 2., 5., 6, 8, 9],
         [3, 4, 7, 8, 10, 11],
         [5, 6, 1, 2, 5, 6],
         [7, 8, 3, 4, 7, 8],
         [8, 9, 5, 6, 1, 2],
         [10, 11, 7, 8, 3, 4]])
    self._test_make_toeplitz_matrix(inputs, output_expected)

  def _test_make_toeplitz_matrix(self, inputs, output_expected):
    output_tf = math_utils.make_toeplitz_matrix(inputs)
    with self.cached_session() as sess:
      output_tf_np = sess.run(output_tf)
    self.assertAllClose(output_tf_np, output_expected)


class TestMakeCovarianceMatrix(test.TestCase):

  def test_zero_size_matrix(self):
    raw = numpy.zeros([0, 0])
    with self.cached_session():
      constructed = math_utils.sign_magnitude_positive_definite(raw=raw).eval()
    self.assertEqual((0, 0), constructed.shape)

  def test_sign_magnitude_positive_definite(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        matrix_tensor = math_utils.sign_magnitude_positive_definite(
            raw=constant_op.constant([[-1., -2.], [3., 4.]], dtype=dtype),
            off_diagonal_scale=constant_op.constant(-1., dtype=dtype),
            overall_scale=constant_op.constant(1., dtype=dtype))
        matrix_evaled = matrix_tensor.eval()
        self.assertAllClose(matrix_evaled, matrix_evaled.T)
        self.assertTrue(numpy.all(numpy.linalg.eigvals(matrix_evaled) > 0))


class TestLookupTable(test.TestCase):

  def test_tuple_of_tensors_lookup(self):
    hash_table = math_utils.TupleOfTensorsLookup(
        key_dtype=dtypes.int64,
        default_values=[[
            array_ops.ones([3, 2], dtype=dtypes.float32),
            array_ops.zeros([5], dtype=dtypes.float64)
        ],
                        array_ops.ones([7, 7], dtype=dtypes.int64)],
        empty_key=-1,
        deleted_key=-2,
        name="test_lookup")
    def stack_tensor(base_tensor):
      return array_ops.stack([base_tensor + 1, base_tensor + 2])

    with self.cached_session() as session:
      ((float_output, double_output), int_output) = session.run(
          hash_table.lookup([2, 1, 0]))
      def expected_output_before_insert(base_tensor):
        return [base_tensor,
                base_tensor,
                base_tensor]
      self.assertAllClose(
          expected_output_before_insert(numpy.ones([3, 2])),
          float_output)
      self.assertAllClose(
          expected_output_before_insert(numpy.zeros([5])),
          double_output)
      self.assertAllEqual(
          expected_output_before_insert(numpy.ones([7, 7], dtype=numpy.int64)),
          int_output)
      hash_table.insert(
          keys=[1, 2],
          values=[[
              stack_tensor(array_ops.ones([3, 2], dtype=dtypes.float32)),
              stack_tensor(array_ops.zeros([5], dtype=dtypes.float64))
          ], stack_tensor(array_ops.ones([7, 7], dtype=dtypes.int64))]).run()
      ((float_output, double_output), int_output) = session.run(
          hash_table.lookup([2, 1, 0]))
      def expected_output_after_insert(base_tensor):
        return [base_tensor + 2,
                base_tensor + 1,
                base_tensor]
      self.assertAllClose(
          expected_output_after_insert(numpy.ones([3, 2])),
          float_output)
      self.assertAllClose(
          expected_output_after_insert(numpy.zeros([5])),
          double_output)
      self.assertAllEqual(
          expected_output_after_insert(numpy.ones([7, 7], dtype=numpy.int64)),
          int_output)


class InputStatisticsTests(test.TestCase):

  def _input_statistics_test_template(
      self, stat_object, num_features, dtype, give_full_data,
      warmup_iterations=0, rtol=1e-6, data_length=500, chunk_size=4):
    graph = ops.Graph()
    with graph.as_default():
      numpy_dtype = dtype.as_numpy_dtype
      values = (
          (numpy.arange(data_length, dtype=numpy_dtype)[..., None]
           + numpy.arange(num_features, dtype=numpy_dtype)[None, ...])[None])
      times = 2 * (numpy.arange(data_length)[None]) - 3
      if give_full_data:
        stat_object.set_data((times, values))
      features = {TrainEvalFeatures.TIMES: times,
                  TrainEvalFeatures.VALUES: values}
      input_fn = input_pipeline.RandomWindowInputFn(
          batch_size=16, window_size=chunk_size,
          time_series_reader=input_pipeline.NumpyReader(features))
      statistics = stat_object.initialize_graph(
          features=input_fn()[0])
      with self.session(graph=graph) as session:
        variables.global_variables_initializer().run()
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(session, coord=coordinator)
        for _ in range(warmup_iterations):
          # A control dependency should ensure that, for queue-based statistics,
          # a use of any statistic is preceded by an update of all adaptive
          # statistics.
          statistics.total_observation_count.eval()
        self.assertAllClose(
            range(num_features) + numpy.mean(numpy.arange(chunk_size))[None],
            statistics.series_start_moments.mean.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.tile(numpy.var(numpy.arange(chunk_size))[None],
                       [num_features]),
            statistics.series_start_moments.variance.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.mean(values[0], axis=0),
            statistics.overall_feature_moments.mean.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.var(values[0], axis=0),
            statistics.overall_feature_moments.variance.eval(),
            rtol=rtol)
        self.assertAllClose(
            -3,
            statistics.start_time.eval(),
            rtol=rtol)
        self.assertAllClose(
            data_length,
            statistics.total_observation_count.eval(),
            rtol=rtol)
        coordinator.request_stop()
        coordinator.join()

  def test_queue(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      for num_features in [1, 2, 3]:
        self._input_statistics_test_template(
            math_utils.InputStatisticsFromMiniBatch(
                num_features=num_features, dtype=dtype),
            num_features=num_features,
            dtype=dtype,
            give_full_data=False,
            warmup_iterations=1000,
            rtol=0.1)


if __name__ == "__main__":
  test.main()
