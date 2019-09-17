# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision.experimental import policy as mp_policy
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision


@test_util.run_all_in_graph_and_eager_modes
class PolicyTest(test.TestCase):
  """Tests Policies."""

  @testing_utils.enable_v2_dtype_behavior
  def test_dtype_attributes(self):
    policy = mp_policy.Policy('infer')
    self.assertEqual(policy.compute_dtype, None)
    self.assertEqual(policy.variable_dtype, None)

    policy = mp_policy.Policy('infer_float32_vars')
    self.assertEqual(policy.compute_dtype, None)
    self.assertEqual(policy.variable_dtype, 'float32')

    for dtype in 'int32', 'bool', 'float16', 'float32':
      policy = mp_policy.Policy(dtype)
      self.assertEqual(policy.name, dtype)
      self.assertEqual(policy.compute_dtype, dtype)
      self.assertEqual(policy.variable_dtype, dtype)

      policy = mp_policy.Policy(dtype + '_with_float32_vars')
      expected_name = (
          dtype if dtype == 'float32' else dtype + '_with_float32_vars')
      self.assertEqual(policy.name, expected_name)
      self.assertEqual(policy.compute_dtype, dtype)
      self.assertEqual(policy.variable_dtype, 'float32')

    for dtype in 'float16', 'bfloat16':
      policy = mp_policy.Policy('mixed_' + dtype)
      self.assertEqual(policy.name, 'mixed_' + dtype)
      self.assertEqual(policy.compute_dtype, dtype)
      self.assertEqual(policy.variable_dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_repr(self):
    for policy in ('infer', 'infer_with_float32_vars', 'float32',
                   'float16_with_float32_vars'):
      self.assertEqual(repr(mp_policy.Policy(policy)),
                       '<Policy "%s", loss_scale=None>' % policy)
    self.assertEqual(repr(mp_policy.Policy('float32_with_float32_vars')),
                     '<Policy "float32", loss_scale=None>')
    self.assertEqual(repr(mp_policy.Policy('float16', loss_scale=2)),
                     '<Policy "float16", loss_scale=FixedLossScale(2.0)>')

  @testing_utils.enable_v2_dtype_behavior
  def test_policy_errors(self):
    # Test passing invalid strings
    expected_error = 'Cannot convert value %s to a mixed precision Policy.'

    for invalid_policy in ('abc', 'abc_with_float32_vars',
                           'float32_with_float16_vars'):
      with self.assertRaisesRegexp(ValueError,
                                   expected_error % invalid_policy):
        mp_policy.Policy(invalid_policy)

    # Test passing a DType
    with self.assertRaisesRegexp(TypeError,
                                 "'name' must be a string, not a DType. "
                                 "Instead, pass DType.name. Got: float16"):
      mp_policy.Policy(dtypes.float16)

    # Test passing a non-DType invalid type
    with self.assertRaisesRegexp(TypeError,
                                 "'name' must be a string, but got: 5"):
      mp_policy.Policy(5)

  @testing_utils.enable_v2_dtype_behavior
  def test_with_input_dtype(self):
    policy = mp_policy.with_input_dtype(mp_policy.Policy('infer'), 'float16')
    self.assertEqual(policy.compute_dtype, 'float16')
    self.assertEqual(policy.variable_dtype, 'float16')

    policy = mp_policy.with_input_dtype(
        mp_policy.Policy('infer_with_float32_vars'), 'float16')
    self.assertEqual(policy.compute_dtype, 'float16')
    self.assertEqual(policy.variable_dtype, 'float32')

    policy = mp_policy.with_input_dtype(
        mp_policy.Policy('infer_with_float32_vars'), 'float32')
    self.assertEqual(policy.compute_dtype, 'float32')
    self.assertEqual(policy.variable_dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_loss_scale(self):
    policy = mp_policy.Policy('float32')
    self.assertEqual(policy.loss_scale, None)

    policy = mp_policy.Policy('float32', loss_scale=None)
    self.assertEqual(policy.loss_scale, None)

    ls = loss_scale_module.DynamicLossScale()
    policy = mp_policy.Policy('float32', loss_scale=ls)
    self.assertIs(policy.loss_scale, ls)

    policy = mp_policy.Policy('float32', loss_scale='dynamic')
    self.assertIsInstance(policy.loss_scale, loss_scale_module.DynamicLossScale)

    policy = mp_policy.Policy('mixed_float16')
    self.assertIsInstance(policy.loss_scale, loss_scale_module.DynamicLossScale)

    policy = mp_policy.Policy('mixed_float16', loss_scale=None)
    self.assertEqual(policy.loss_scale, None)

    policy = mp_policy.Policy('mixed_bfloat16')
    self.assertEqual(policy.loss_scale, None)

  @testing_utils.enable_v2_dtype_behavior
  def test_global_policy(self):
    if base_layer_utils.v2_dtype_behavior_enabled():
      default_policy = 'float32'
    else:
      default_policy = 'infer'
    self.assertEqual(mp_policy.global_policy().name, default_policy)
    try:
      mp_policy.set_policy('infer_with_float32_vars')
      self.assertEqual(mp_policy.global_policy().name,
                       'infer_with_float32_vars')
      with ops.Graph().as_default():  # Policies are not associated with a graph
        self.assertEqual(mp_policy.global_policy().name,
                         'infer_with_float32_vars')
      mp_policy.set_policy('infer')
      self.assertEqual(mp_policy.global_policy().name, 'infer')
      policy = mp_policy.Policy('infer_with_float32_vars')
      mp_policy.set_policy(policy)
      self.assertIs(mp_policy.global_policy(), policy)
    finally:
      mp_policy.set_policy(None)

  @testing_utils.enable_v2_dtype_behavior
  def test_loss_scale_warning(self):
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      mp_policy.Policy('float32', loss_scale=2.)
      self.assertEqual(
          mock_warn.call_args[0][0],
          'Creating a Policy with a loss scale is only useful for float16 '
          'policies. You passed loss_scale=2.0 for policy float32. Consider '
          'not passing any loss_scale instead.')

    for policy_name in 'float16', 'mixed_float16':
      with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
        mp_policy.Policy(policy_name, loss_scale=2.)
        mock_warn.assert_not_called()

  @testing_utils.enable_v2_dtype_behavior
  def test_float32_vars_warning(self):
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      mp_policy.Policy('infer_with_float32_vars')
      self.assertEqual(
          mock_warn.call_args[0][0],
          "WARNING: The 'infer_with_float32_vars' policy is deprecated and "
          "will be removed in TensorFlow 2.1. Please use the 'mixed_float16' "
          "or 'mixed_bfloat16' policy instead.")

    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      mp_policy.Policy('float16_with_float32_vars')
      self.assertEqual(
          mock_warn.call_args[0][0],
          "WARNING: The 'float16_with_float32_vars' policy is deprecated and "
          "will be removed in TensorFlow 2.1. Please use the 'mixed_float16' "
          "policy instead.")

    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      mp_policy.Policy('bfloat16_with_float32_vars')
      self.assertEqual(
          mock_warn.call_args[0][0],
          "WARNING: The 'bfloat16_with_float32_vars' policy is deprecated and "
          "will be removed in TensorFlow 2.1. Please use the 'mixed_bfloat16' "
          "policy instead.")

    with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
      mp_policy.Policy('float64_with_float32_vars')
      self.assertEqual(
          mock_warn.call_args[0][0],
          "WARNING: The 'float64_with_float32_vars' policy is deprecated and "
          "will be removed in TensorFlow 2.1.")

    for policy_name in 'float16', 'float32', 'mixed_float16', 'mixed_bfloat16':
      with test.mock.patch.object(tf_logging, 'warn') as mock_warn:
        mp_policy.Policy(policy_name)
        mock_warn.assert_not_called()

  @testing_utils.enable_v2_dtype_behavior
  def test_policy_scope(self):
    if base_layer_utils.v2_dtype_behavior_enabled():
      default_policy = 'float32'
    else:
      default_policy = 'infer'
    with mp_policy.policy_scope('infer_with_float32_vars'):
      self.assertEqual(mp_policy.global_policy().name,
                       'infer_with_float32_vars')
      with mp_policy.policy_scope('infer'):
        self.assertEqual(mp_policy.global_policy().name, 'infer')
      self.assertEqual(mp_policy.global_policy().name,
                       'infer_with_float32_vars')
    self.assertEqual(mp_policy.global_policy().name, default_policy)

  @testing_utils.enable_v2_dtype_behavior
  def test_error_if_graph_rewrite_enabled(self):
    try:
      mixed_precision.enable_mixed_precision_graph_rewrite(
          gradient_descent.SGD(1.))
      with self.assertRaisesRegexp(
          ValueError, 'the mixed precision graph rewrite has already been '
                      'enabled'):
        mp_policy.set_policy('infer_float32_vars')
    finally:
      mixed_precision.disable_mixed_precision_graph_rewrite()

  @testing_utils.disable_v2_dtype_behavior
  def test_v1_dtype_behavior(self):
    # These policies are allowed with V1 dtype behavior
    with mp_policy.policy_scope(mp_policy.Policy('infer')):
      pass
    with mp_policy.policy_scope(mp_policy.Policy('infer_float32_vars')):
      pass

    # These policies are not allowed with V1 dtype behavior
    with self.assertRaisesRegexp(
        ValueError,
        'global policy can only be set to a non-infer policy in TensorFlow 2'):
      with mp_policy.policy_scope(mp_policy.Policy('float32')):
        pass
    with self.assertRaisesRegexp(
        ValueError,
        'global policy can only be set to a non-infer policy in TensorFlow 2'):
      with mp_policy.policy_scope(
          mp_policy.Policy('float16_with_float32_vars')):
        pass


if __name__ == '__main__':
  test.main()
