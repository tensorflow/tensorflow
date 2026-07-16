# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras advanced activation layers."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ELUValidationTest(test.TestCase):
  """Tests for ELU layer alpha validation."""

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_positive_scalar(self):
    """ELU layer should accept non-negative scalar alpha."""
    advanced_activations.ELU(alpha=0.0)
    advanced_activations.ELU(alpha=1.0)
    advanced_activations.ELU(alpha=0.5)

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_negative_scalar_raises(self):
    """ELU layer should raise ValueError for negative scalar alpha."""
    with self.assertRaisesRegex(
        ValueError, 'should be >= 0'):
      advanced_activations.ELU(alpha=-1.0)

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_tensor_positive(self):
    """ELU layer should accept positive tf.Tensor alpha."""
    advanced_activations.ELU(alpha=constant_op.constant(1.0))

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_variable_positive(self):
    """ELU layer should accept positive tf.Variable alpha."""
    advanced_activations.ELU(alpha=variables.Variable(1.0))

  @test_util.run_deprecated_v1
  def test_elu_alpha_tensor_negative_fails_runtime(self):
    """ELU layer with negative tf.Tensor alpha should fail at runtime."""
    with ops.Graph().as_default():
      layer = advanced_activations.ELU(
          alpha=constant_op.constant(-1.0))
      with self.session() as sess:
        with self.assertRaisesRegex(
            errors.InvalidArgumentError, 'should be >= 0'):
          sess.run(variables.global_variables_initializer())
          # Access self.alpha to trigger the assert op
          sess.run(layer.alpha)

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_tensor_zero(self):
    """ELU layer should accept alpha=0.0 as tf.Tensor."""
    advanced_activations.ELU(alpha=constant_op.constant(0.0))

  @test_util.run_all_in_graph_and_eager_modes
  def test_elu_alpha_none_raises(self):
    """ELU layer should raise ValueError for alpha=None."""
    with self.assertRaisesRegex(ValueError, 'cannot be None'):
      advanced_activations.ELU(alpha=None)


if __name__ == '__main__':
  test.main()
