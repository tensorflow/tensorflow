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
"""Regression tests for negative-penalty rejection in `regularizers.py`.

Covers the fix adding a non-negativity check to `_check_penalty_number`.
Before the fix, a negative `l1`/`l2` coefficient was silently accepted
and produced a *negative* regularization loss -- rewarding larger
weights instead of penalizing them, which defeats the purpose of
regularization. After the fix, constructing any regularizer with a
negative coefficient should raise `ValueError` immediately, matching
the current standalone `keras` package's behavior.
"""

from absl.testing import parameterized

from tensorflow.python.keras import regularizers
from tensorflow.python.platform import test


class NegativePenaltyRejectionTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('l1_small_negative', -0.5),
      ('l1_large_negative', -100.0),
      ('l1_negative_int', -1),
  )
  def test_l1_rejects_negative_penalty(self, l1_value):
    with self.assertRaisesRegex(
        ValueError, 'not a valid regularization penalty number'):
      regularizers.L1(l1=l1_value)

  @parameterized.named_parameters(
      ('l2_small_negative', -0.1),
      ('l2_large_negative', -100.0),
      ('l2_negative_int', -1),
  )
  def test_l2_rejects_negative_penalty(self, l2_value):
    with self.assertRaisesRegex(
        ValueError, 'not a valid regularization penalty number'):
      regularizers.L2(l2=l2_value)

  @parameterized.named_parameters(
      ('negative_l1_only', -0.5, 0.1),
      ('negative_l2_only', 0.1, -0.5),
      ('both_negative', -0.5, -0.1),
  )
  def test_l1l2_rejects_negative_penalty(self, l1_value, l2_value):
    with self.assertRaisesRegex(
        ValueError, 'not a valid regularization penalty number'):
      regularizers.L1L2(l1=l1_value, l2=l2_value)

  @parameterized.named_parameters(
      ('l1_zero', 0.0),
      ('l1_zero_int', 0),
      ('l1_small_positive', 0.01),
  )
  def test_l1_still_accepts_non_negative_penalty(self, l1_value):
    # Regression guard: the fix must not reject the zero/positive cases
    # that were always valid, especially zero, since L1L2's own
    # constructor defaults to l1=0., l2=0.
    reg = regularizers.L1(l1=l1_value)
    self.assertEqual(float(reg.l1), float(l1_value))

  @parameterized.named_parameters(
      ('l2_zero', 0.0),
      ('l2_zero_int', 0),
      ('l2_small_positive', 0.01),
  )
  def test_l2_still_accepts_non_negative_penalty(self, l2_value):
    reg = regularizers.L2(l2=l2_value)
    self.assertEqual(float(reg.l2), float(l2_value))

  def test_l1l2_still_accepts_default_zero_penalties(self):
    # L1L2()'s documented default is l1=0., l2=0. -- must still construct
    # without raising after the fix.
    reg = regularizers.L1L2()
    self.assertEqual(float(reg.l1), 0.0)
    self.assertEqual(float(reg.l2), 0.0)


if __name__ == '__main__':
  test.main()
