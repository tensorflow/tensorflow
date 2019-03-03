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

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.mixed_precision.experimental import policy as mp_policy
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PolicyTest(test.TestCase):
  """Tests Policies."""

  def test_infer(self):
    policy = mp_policy.Policy('infer')
    self.assertEqual(policy.name, 'infer')
    self.assertEqual(policy.default_variable_dtype, None)

  def test_infer_float32_vars(self):
    policy = mp_policy.Policy('infer_float32_vars')
    self.assertEqual(policy.name, 'infer_float32_vars')
    self.assertEqual(policy.default_variable_dtype, 'float32')

  def test_global_policy(self):
    self.assertEqual(mp_policy.global_policy().name, 'infer')
    default_policy = mp_policy.global_policy()
    try:
      mp_policy.set_policy('infer_float32_vars')
      self.assertEqual(mp_policy.global_policy().name, 'infer_float32_vars')
      self.assertEqual(mp_policy.global_policy().default_variable_dtype,
                       'float32')
      with ops.Graph().as_default():  # Policies are not associated with a graph
        self.assertEqual(mp_policy.global_policy().name, 'infer_float32_vars')
      mp_policy.set_policy('infer')
      self.assertEqual(mp_policy.global_policy().name, 'infer')
      self.assertEqual(mp_policy.global_policy().default_variable_dtype, None)
      policy = mp_policy.Policy('infer_float32_vars')
      mp_policy.set_policy(policy)
      self.assertIs(mp_policy.global_policy(), policy)
    finally:
      mp_policy.set_policy(default_policy)

  def test_policy_scope(self):
    with mp_policy.policy_scope('infer_float32_vars'):
      self.assertEqual(mp_policy.global_policy().name, 'infer_float32_vars')
      with mp_policy.policy_scope('infer'):
        self.assertEqual(mp_policy.global_policy().name, 'infer')
      self.assertEqual(mp_policy.global_policy().name, 'infer_float32_vars')
    self.assertEqual(mp_policy.global_policy().name, 'infer')

if __name__ == '__main__':
  test.main()
