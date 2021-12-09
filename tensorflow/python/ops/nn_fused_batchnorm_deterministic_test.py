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
"""Functional tests for fused batch-norm related to determinism."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_impl
# The following import is required to register the gradient function
from tensorflow.python.ops.nn_grad import _FusedBatchNormV3Grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class FusedBatchNormalizationDeterministicTest(test.TestCase,
                                               parameterized.TestCase):
  """Test determinsitic functionality and exceptions for FusedBatchNorm.

  Test that tf.errors.UnimplementedError is thrown, as
  appropriate, by the GPU code-path through FusedBatchNormFreezeGrad when
  deterministic ops are enabled. This test assumes that
  nn_fused_batchnorm_test.py runs equivalent test cases when deterministic ops
  are not enabled and will therefore detect erroneous exception throwing in
  those cases.

  Also test that the other code-paths, running on both CPU and GPU, operate
  deterministically.
  """

  def _genParams(self, data_format, x_dtype, large_batch):
    if large_batch:
      batch_size = 5000
      height = width = 4
    else:
      batch_size = 10
      height = 5
      width = 5000
    channel_count = 3
    if data_format == 'NHWC':
      x_shape = (batch_size, height, width, channel_count)
    else:  # 'NCHW'
      x_shape = (batch_size, channel_count, height, width)
    # Using random_ops.random_normal would produce different values on each run
    x = constant_op.constant(np.random.normal(size=x_shape), dtype=x_dtype)
    scale_shape = (channel_count,)
    scale = constant_op.constant(
        np.random.normal(size=scale_shape), dtype=dtypes.float32)
    offset = constant_op.constant(
        np.random.normal(size=scale_shape), dtype=dtypes.float32)
    mean = np.random.normal(size=scale_shape)
    variance = np.random.normal(size=scale_shape)
    y_shape = x_shape
    y_dtype = x_dtype
    upstream_gradients = constant_op.constant(
        np.random.normal(size=y_shape), dtype=y_dtype)
    return x, scale, offset, mean, variance, upstream_gradients

  @parameterized.parameters('NHWC', 'NCHW')
  def testForward(self, data_format):
    with self.cached_session():
      for large_batch in [False, True]:
        for x_dtype in [dtypes.float16, dtypes.float32]:  # skipping bfloat16
          x, scale, offset, mean, variance, _ = self._genParams(
              data_format, x_dtype, large_batch)
          for is_training in [False, True]:
            op_output = nn_impl.fused_batch_norm(
                x,
                scale,
                offset,
                mean,
                variance,
                data_format=data_format,
                is_training=is_training,
                exponential_avg_factor=1.01)
            y_a, running_mean_a, running_var_a = op_output
            y_a = self.evaluate(y_a)
            if is_training:
              running_mean_a = self.evaluate(running_mean_a)
              running_var_a = self.evaluate(running_var_a)
            for _ in range(5):
              op_output_b = nn_impl.fused_batch_norm(
                  x,
                  scale,
                  offset,
                  mean,
                  variance,
                  data_format=data_format,
                  is_training=is_training,
                  exponential_avg_factor=1.01)
              y_b, running_mean_b, running_var_b = op_output_b
              y_b = self.evaluate(y_b)
              self.assertAllEqual(y_a, y_b)
              if is_training:
                running_mean_b = self.evaluate(running_mean_b)
                running_var_b = self.evaluate(running_var_b)
                self.assertAllEqual(running_mean_a, running_mean_b)
                self.assertAllEqual(running_var_a, running_var_b)

  @parameterized.parameters('NHWC', 'NCHW')
  @test_util.disable_xla('XLA is deterministic')
  def testBackward(self, data_format):
    with self.cached_session():
      for large_batch in [False, True]:
        # Only run with float32, as float16 is very slow on CPUs
        params = self._genParams(data_format, dtypes.float32, large_batch)
        x, scale, offset, mean, variance, upstream_gradients = params
        for is_training in [False, True]:
          for backprop_to in [x, scale, offset]:
            with backprop.GradientTape(persistent=True) as tape:
              tape.watch(backprop_to)
              op_output = nn_impl.fused_batch_norm(
                  x,
                  scale,
                  offset,
                  mean,
                  variance,
                  data_format=data_format,
                  is_training=is_training,
                  exponential_avg_factor=0.99)
              gradient_injector_output = op_output[0] * upstream_gradients
            if (len(config.list_physical_devices('GPU')) and
                not is_training):
              # Only backprop to offset is nondeterministic (on GPU, when
              # is_training=False), but backprop to the other parameters is
              # calculated using the same kernel.
              with self.assertRaisesRegex(
                  errors_impl.UnimplementedError,
                  'A deterministic GPU implementation of fused batch-norm' +
                  ' backprop, when training is disabled, is not currently' +
                  ' available.'):
                grad = tape.gradient(gradient_injector_output, backprop_to)
                self.evaluate(grad)
            else:
              grad_a = tape.gradient(gradient_injector_output, backprop_to)
              grad_a = self.evaluate(grad_a)
              for _ in range(3):
                grad_b = tape.gradient(gradient_injector_output,
                                       backprop_to)
                grad_b = self.evaluate(grad_b)
                self.assertAllEqual(grad_a, grad_b)


if __name__ == '__main__':
  # TODO(reedwm): Merge this file with nn_fused_batchnorm_test.py
  config.enable_op_determinism()
  test.main()
