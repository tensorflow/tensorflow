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
      self.assertEqual(policy.compute_dtype, dtype)
      self.assertEqual(policy.variable_dtype, dtype)

      policy = mp_policy.Policy(dtype + '_with_float32_vars')
      self.assertEqual(policy.compute_dtype, dtype)
      self.assertEqual(policy.variable_dtype, 'float32')

  @testing_utils.enable_v2_dtype_behavior
  def test_repr(self):
    for policy in ('infer', 'infer_with_float32_vars', 'float32',
                   'float16_with_float32_vars'):
      self.assertEqual(repr(mp_policy.Policy(policy)),
                       '<Policy "%s">' % policy)
    self.assertEqual(repr(mp_policy.Policy('float32_with_float32_vars')),
                     '<Policy "float32">')

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
