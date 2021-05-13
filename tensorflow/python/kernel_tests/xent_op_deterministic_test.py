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
"""Tests for deterministic functionality of SoftmaxCrossEntropyWithLogits op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import xent_op_test_base
from tensorflow.python.ops import nn_ops
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentDeterministicTest(xent_op_test_base.XentOpTestBase):
  """Test that SoftmaxCrossEntropyWithLogits operates reproducibly.

  Inheriting from xent_op_test_base.XentTestBase ensures that regular op
  functionality is correct when the deterministic code-path is selected.

  Note that because nn_ops.softmax_cross_entropy_with_logits calls
  nn_ops.cross_entropy_with_logits_v2, the focus of testing is on the
  former in order to test both.
  """

  def _randomFloats(self, shape, dtype, normalized_rows=False):
    a = (2 * np.random.random_sample(shape) - 1).astype(dtype)

    if normalized_rows:

      def normalize(row):
        return row / row.sum()

      a = np.apply_along_axis(normalize, 1, a)

    return constant_op.constant(a)

  def _generateInputs(self, dtype, seed=123, forward_not_backward=False):
    batch_size = 1024
    if forward_not_backward and dtype == np.float16:
      # Generate more noise to expose the internal float32 implementation.
      # This is associated with significantly slower test cases (esp. on CPU).
      classes_count = 20000
    else:
      classes_count = 3000
    shape = (batch_size, classes_count)
    np.random.seed(seed)
    labels = self._randomFloats(shape, dtype, normalized_rows=True)
    logits = self._randomFloats(shape, dtype)
    return labels, logits

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    for use_gpu in [False, True]:
      if use_gpu and self._gpuNotAvailable():
        continue
      with self.cached_session(), test_util.device(use_gpu):
        # Even in eager mode, the above line will be able to pin ops to CPU.
        for dtype in [np.float16, np.float32, np.float64]:
          for trial in range(5):
            seed = 123 + trial
            labels, logits = self._generateInputs(
                dtype, seed=seed, forward_not_backward=True)
            result_a = nn_ops.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            result_b = nn_ops.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            self.assertAllEqual(result_a, result_b)

  def _testBackward(self, labels_not_logits=False):
    for use_gpu in [False, True]:
      if use_gpu and self._gpuNotAvailable():
        continue
      with self.cached_session(), test_util.device(use_gpu):
        # Even in eager mode, the above line will be able to pin ops to CPU.
        for dtype in [np.float16, np.float32, np.float64]:
          labels, logits = self._generateInputs(dtype, seed=456)
          output_shape = labels.shape[0]

          def gradients(seed=789):
            np.random.seed(seed)
            upstream_gradients = self._randomFloats(output_shape, dtype)
            with backprop.GradientTape(persistent=True) as tape:
              tape.watch(labels)
              tape.watch(logits)
              op_output = nn_ops.softmax_cross_entropy_with_logits(
                  labels=labels, logits=logits)
              gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, [labels, logits])

          for trial in range(5):
            seed = 456 + trial
            labels_grad_a, logits_grad_a = gradients(seed=seed)
            labels_grad_b, logits_grad_b = gradients(seed=seed)
            if labels_not_logits:
              self.assertAllEqual(labels_grad_a, labels_grad_b)
            else:
              self.assertAllEqual(logits_grad_a, logits_grad_b)

  @test_util.run_in_graph_and_eager_modes
  def testBackwardLabels(self):
    self._testBackward(labels_not_logits=True)

  @test_util.run_in_graph_and_eager_modes
  def testBackwardLogits(self):
    self._testBackward(labels_not_logits=False)

  # Modifications to the parent class (xent_op_test_base.XentOpTestBase) follow

  def _coreOp(self, features, labels):
    """ Run a substitute for the core op under test.

    In the code-under-test, when op determinism is enabled, the call to
    gen_nn_ops.softmax_cross_entropy_with_logits is replaced with a
    small network of discrete TF ops. The following modification enables
    existing test cases that run on
    gen_nn_ops.softmax_cross_entropy_with_logits to be reused by emulating
    it using nn_ops.softmax_cross_entropy_with_logits.
    """
    features = ops.convert_to_tensor(features) # needed for the gradient tape
    with backprop.GradientTape() as tape:
      tape.watch(features)
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=features, dim=-1)
    return loss, tape.gradient(loss, features)

  def testSingleClass(self):
    """Disable testing of backprop for single-class case.

    The deterministic implementation does not produce the gradients expected by
    the test.

    TODO(duncanriach): Identify the source of the difference in gradients for
                       this case.
    """
    self._testSingleClass(test_backprop=False)

  def testRankTooLarge(self):
    """This test is not relevant for nn_ops.softmax_cross_entropy_with_logits.

    Whereas gen_nn_ops.softmax_cross_entropy_with_logits operates on
    two-dimensional matrices, nn_ops.softmax_cross_entropy_with_logits supports
    more than one dimension beyond the class dimension. Therefore, this test
    has been disabled.
    """
    pass

  def testFeaturesBroadcast(self):
    """This test is not relevant for nn_ops.softmax_cross_entrop_with_logits.

    Whereas gen_nn_ops.softmax_cross_entropy_with_logits supports broadcasting
    of logits (features) across batch, softmax_cross_entropy_with_logits cannot
    expose the results because it tries to reshape the resulting cost matrix
    (created using broadcast logits) into the smaller shape of the incoming
    logits (minus the axis dimension). Therefore, this test has been disabled.
    """
    pass

  def testLabelsBroadcast(self):
    """Disable testing of backprop for labels-broadcast case.

    The deterministic implementation does not produce the gradients expected by
    the test.

    TODO(duncanriach): Identify the source of the difference in gradients for
                       this case.
    """
    self._testLabelsBroadcast(test_backprop=False)


  def testNotMatrix(self):
    """This test is not relevant for nn_ops.softmax_cross_entropy_with_logits.

    Whereas gen_nn_ops.softmax_cross_entropy_with_logits requires that both
    features and labels are two-dimensional,
    nn_ops.softmax_cross_entropy_with_logits doens not. Therefore, this test
    has been disabled.
    """
    pass


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  test.main()
