# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for logical module."""

from tensorflow.python.autograph.operators import logical
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class LogicalOperatorsTest(test.TestCase):

  def assertNotCalled(self):
    self.fail('this should not be called')

  def _tf_true(self):
    return constant_op.constant(True)

  def _tf_false(self):
    return constant_op.constant(False)

  def test_and_python(self):
    self.assertTrue(logical.and_(lambda: True, lambda: True))
    self.assertTrue(logical.and_(lambda: [1], lambda: True))
    self.assertListEqual(logical.and_(lambda: True, lambda: [1]), [1])

    self.assertFalse(logical.and_(lambda: False, lambda: True))
    self.assertFalse(logical.and_(lambda: False, self.assertNotCalled))

  @test_util.run_deprecated_v1
  def test_and_tf(self):
    with self.cached_session() as sess:
      t = logical.and_(self._tf_true, self._tf_true)
      self.assertEqual(self.evaluate(t), True)
      t = logical.and_(self._tf_true, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      t = logical.and_(self._tf_false, lambda: True)
      self.assertEqual(self.evaluate(t), False)
      # TODO(mdan): Add a test for ops with side effects.

  def test_or_python(self):
    self.assertFalse(logical.or_(lambda: False, lambda: False))
    self.assertFalse(logical.or_(lambda: [], lambda: False))
    self.assertListEqual(logical.or_(lambda: False, lambda: [1]), [1])

    self.assertTrue(logical.or_(lambda: False, lambda: True))
    self.assertTrue(logical.or_(lambda: True, self.assertNotCalled))

  @test_util.run_deprecated_v1
  def test_or_tf(self):
    with self.cached_session() as sess:
      t = logical.or_(self._tf_false, self._tf_true)
      self.assertEqual(self.evaluate(t), True)
      t = logical.or_(self._tf_false, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      t = logical.or_(self._tf_true, lambda: True)
      self.assertEqual(self.evaluate(t), True)
      # TODO(mdan): Add a test for ops with side effects.

  def test_not_python(self):
    self.assertFalse(logical.not_(True))
    self.assertFalse(logical.not_([1]))
    self.assertTrue(logical.not_([]))

  def test_not_tf(self):
    with self.cached_session() as sess:
      t = logical.not_(self._tf_false())
      self.assertEqual(self.evaluate(t), True)


if __name__ == '__main__':
  test.main()
