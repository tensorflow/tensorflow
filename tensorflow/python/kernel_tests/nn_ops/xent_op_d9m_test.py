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

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import xent_op_test_base
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentOpDeterminismExceptionsTest(test.TestCase):
  """Test d9m-unimplemented exceptions from SoftmaxXentWithLogitsOp.

  Test that tf.errors.UnimplementedError is thrown, as appropriate, by the GPU
  code-paths through SoftmaxXentWithLogitsOp when deterministic ops are
  enabled.

  This test assumes that xent_op_test.py runs equivalent test cases when
  deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testExceptionThrowing(self):
    with self.session(), test_util.force_gpu():
      for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
        features = constant_op.constant([[0.3, 0.5], [0.5, 0.6]], dtype=dtype)
        labels = constant_op.constant([[0.2, 0.4], [0.1, 0.2]], dtype=dtype)
        with self.assertRaisesRegex(
            errors_impl.UnimplementedError,
            "The GPU implementation of SoftmaxCrossEntropyWithLogits that " +
            "would have been executed is not deterministic. Note that the " +
            "Python API uses an alternative, deterministic, GPU-accelerated " +
            "path when determinism is enabled."):
          result = gen_nn_ops.softmax_cross_entropy_with_logits(
              features=features, labels=labels)
          self.evaluate(result)


class XentOpDeterministicTest(xent_op_test_base.XentOpTestBase):
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
    with self.cached_session():
      for dtype in [np.float16, np.float32, np.float64,  \
        dtypes.bfloat16.as_numpy_dtype]:

        for trial in range(5):
          seed = 123 + trial
          labels, logits = self._generateInputs(
              dtype, seed=seed, forward_not_backward=True)
          result_a = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
          result_b = nn_ops.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
          self.assertAllEqual(result_a, result_b)

  @test_util.run_in_graph_and_eager_modes
  def testBackward(self):
    with self.cached_session():
      for dtype in [np.float16, np.float32, np.float64,  \
        dtypes.bfloat16.as_numpy_dtype]:
        labels, logits = self._generateInputs(dtype, seed=456)
        output_shape = labels.shape[0]

        def gradients(seed):
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
          self.assertAllEqual(labels_grad_a, labels_grad_b)
          self.assertAllEqual(logits_grad_a, logits_grad_b)

  # Modifications to the parent class (xent_op_test_base.XentOpTestBase) follow

  def testSingleClass(self):
    """Modify testing of gradient for single-class case.

    The deterministic implementation does not produce the gradients expected by
    the original test (for the nondeterministic functionality) when the labels
    vector is not a valid probability distribution.

    labels: [[-1.], [0.], [1.], [1.]]
    logits: [[1.], [-1.], [0.], [1.]]

                   nondeterministic               deterministic
    dloss/dlogits: [[2.0], [1.0], [0.0], [0.0]]   [[0.0], [0.0], [0.0], [0.0]]

    Note that only the second two label vectors are valid probability
    distributions (as required by the API) and that the gradient matches for
    those cases.

    TODO(duncanriach): Further investigate the source of the difference in
                       the gradients for this case.
    """
    self._testSingleClass(expected_gradient=[[0.0], [0.0], [0.0], [0.0]])

  def testLabelsBroadcast(self):
    """Modify testing of gradient for labels-broadcast case.

    The deterministic implementation does not produce the gradients expected by
    the original test (for the nondeterministic functionality) when the labels
    vector (after broadcasting) is not a valid probability distribution.

    labels: [[0.], [2.], [0.25]]
    logits: [[1., 1., 1., 1.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]]

    dloss/dlogits (nondeterministic):
        [[ 0.25 ,  0.25 ,  0.25 ,  0.25 ],
         [-1.968, -1.913, -1.763, -1.355],
         [-0.218, -0.163, -0.013,  0.394]]

    dloss/dlogits (determinsitic):
        [[ 0.   ,  0.   ,  0.   ,  0.   ],
         [-1.743, -1.303, -0.105,  3.150],
         [-0.218, -0.163, -0.013,  0.394]]

    Note that neither of the first two broadcast label vectors is a valid
    probability distribution (as required by the API) and that these are the
    cases that yield different gradients for nondeterministic vs determinsitic
    implementations.

    TODO(duncanriach): Further investigate the source of the difference in
                       the gradient for this case.
    """
    self._testLabelsBroadcast(uniform_labels_gradient=[[
        0., 0., 0., 0.
    ], [-1.743, -1.303, -0.105, 3.150], [-0.218, -0.163, -0.013, 0.394]])


if __name__ == "__main__":
  config.enable_op_determinism()
  test.main()
