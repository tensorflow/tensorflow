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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SparseSoftmaxCrossEntropyWithLogitsDeterminismExceptionsTest(
    test.TestCase):
  """Test d9m-unimplemented exceptions from SparseSoftmaxCrossEntropyWithLogits.

  Test that tf.errors.UnimplementedError is thrown or not thrown, as
  appropriate, by the GPU code-paths for SparseSoftmaxCrossEntropyWithLogits
  when deterministic ops are enabled.

  This test assumes that the base op test runs all the same test cases when
  deterministic ops are not enabled and will therefore detect erroneous
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
            result = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            self.evaluate(result)


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  test.main()
