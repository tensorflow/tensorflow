# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for deterministic functionality of SparseSoftmaxCrossEntropyWithLogits op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SparseSoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SparseSoftmaxCrossEntropyWithLogitsDeterminismExceptionsTest(
    test.TestCase):
  """Test d9m-unimplemented exceptions from SparseSoftmaxCrossEntropyWithLogits.

  Test that tf.errors.UnimplementedError is thrown or not thrown, as
  appropriate, by the GPU code-paths for SparseSoftmaxCrossEntropyWithLogits
  when deterministic ops are enabled.

  This test assumes that sparse_xent_op_test.py runs all the same test cases
  when deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

  @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testExceptionThrowing(self):
    with self.session(force_gpu=True):
      for logits_dtype in [dtypes.float16, dtypes.float32]:
        for labels_dtype in [dtypes.int32, dtypes.int64]:
          labels = constant_op.constant([1, 0], dtype=labels_dtype)
          logits = constant_op.constant([[0.3, 0.5], [0.2, 0.6]],
                                        dtype=logits_dtype)
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError,
              "Deterministic GPU implementation of " +
              "SparseSoftmaxCrossEntropyWithLogits not available."):
            result = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            self.evaluate(result)


class SparseSoftmaxCrossEntropyWithLogitsDeterministicTest(test.TestCase):
  """Test that SparseSoftmaxCrossEntropyWithLogits operates reproducibly.

  Note that the deterministic functionality currently tested by this class is
  always activated (not enabled by TF_DETERMINISTIC_OPS), so this class does not
  currently need to inherit from a base op test class derived from
  sparse_xent_op_test.py (to ensure that the op still functions correctly when
  determinism is enabled).
  """

  def _randomInts(self, shape, high, dtype):
    return constant_op.constant(
        np.random.randint(low=0, high=high, size=shape).astype(dtype))

  def _randomFloats(self, shape, dtype, normalized_rows=False):
    a = (2 * np.random.random_sample(shape) - 1).astype(dtype)

    if normalized_rows:

      def normalize(row):
        return row / row.sum()

      a = np.apply_along_axis(normalize, 1, a)

    return constant_op.constant(a)

  def _generateInputs(self, labels_dtype, logits_dtype, seed=123):
    batch_size = 1024
    classes_count = 1000
    np.random.seed(seed)
    labels_shape = (batch_size)
    labels = self._randomInts(
        labels_shape, high=classes_count, dtype=labels_dtype)
    logits_shape = (batch_size, classes_count)
    logits = self._randomFloats(logits_shape, logits_dtype)
    return labels, logits

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    with self.session(), test_util.force_cpu():
      for logits_dtype in [np.float16, np.float32, np.float64]:
        for labels_dtype in [np.int32, np.int64]:
          for trial in range(5):
            seed = 123 + trial
            labels, logits = self._generateInputs(
                labels_dtype, logits_dtype, seed=seed)
            result_a = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            result_b = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            self.assertAllEqual(result_a, result_b)

  @test_util.run_in_graph_and_eager_modes
  def testBackward(self):
    with self.session(), test_util.force_cpu():
      for logits_dtype in [np.float16, np.float32, np.float64]:
        for labels_dtype in [np.int32, np.int64]:
          labels, logits = self._generateInputs(
              labels_dtype, logits_dtype, seed=456)
          output_shape = labels.shape[0]

          def gradients(seed=789):
            np.random.seed(seed)
            upstream_gradients = self._randomFloats(output_shape, logits_dtype)
            with backprop.GradientTape(persistent=True) as tape:
              tape.watch(logits)
              op_output = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                  labels=labels, logits=logits)
              gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, logits)

          for trial in range(5):
            seed = 456 + trial
            result_a = gradients(seed=seed)
            result_b = gradients(seed=seed)
            self.assertAllEqual(result_a, result_b)


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  test.main()
