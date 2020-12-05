# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Tests for tensorflow_datasets.core.utils.version."""
from tensorflow_datasets import testing
from tensorflow.data.experimental.core.utils import version


class VersionTest(testing.TestCase):

  def test_str_to_version(self):
    self.assertEqual(version._str_to_version('1.2.3'), (1, 2, 3))
    self.assertEqual(version._str_to_version('1.2.*', True), (1, 2, '*'))
    self.assertEqual(version._str_to_version('1.*.3', True), (1, '*', 3))
    self.assertEqual(version._str_to_version('*.2.3', True), ('*', 2, 3))
    self.assertEqual(version._str_to_version('1.*.*', True), (1, '*', '*'))
    with self.assertRaisesWithPredicateMatch(ValueError, 'Invalid version '):
      version.Version('1.3')
    with self.assertRaisesWithPredicateMatch(ValueError, 'Invalid version '):
      version.Version('1.3.*')

  def test_version(self):
    v = version.Version('1.3.534')
    self.assertEqual((v.major, v.minor, v.patch), (1, 3, 534))
    self.assertEqual(str(v), '1.3.534')
    self.assertEqual(repr(v), "Version('1.3.534')")
    with self.assertRaisesWithPredicateMatch(ValueError, 'Format should be '):
      version.Version('1.3.-534')
    with self.assertRaisesWithPredicateMatch(ValueError, 'Format should be '):
      version.Version('1.3')
    with self.assertRaisesWithPredicateMatch(ValueError, 'Format should be '):
      version.Version('1.3.')
    with self.assertRaisesWithPredicateMatch(ValueError, 'Format should be '):
      version.Version('1..5')
    with self.assertRaisesWithPredicateMatch(ValueError, 'Format should be '):
      version.Version('a.b.c')

  def test_comparison(self):
    v = version.Version('1.3.534')
    self.assertLess(v, version.Version('1.3.999'))
    self.assertLess(v, '1.3.999')
    self.assertGreater('1.4.5', v)
    self.assertEqual(v, '1.3.534')
    self.assertNotEqual(v, version.Version('1.3.535'))

  def test_invalid_comparison(self):
    v = version.Version('1.3.534')
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Format should be '):
      unused_ = v < 'abc'
    with self.assertRaisesWithPredicateMatch(
        AssertionError, 'cannot be compared to version'):
      unused_ = v > 123

  def test_match(self):
    v = version.Version('1.2.3')
    self.assertTrue(v.match('1.2.3'))
    self.assertTrue(v.match('1.2.*'))
    self.assertTrue(v.match('1.*.*'))
    self.assertTrue(v.match('*.*.*'))
    self.assertTrue(v.match('*.2.3'))
    self.assertFalse(v.match('1.2.4'))
    self.assertFalse(v.match('1.3.*'))
    self.assertFalse(v.match('1.3.*'))
    self.assertFalse(v.match('2.*.*'))

  def test_eq(self):
    v1 = version.Version('1.2.3')
    v2 = version.Version('1.2.3')
    v3 = '1.2.3'
    # pylint: disable=g-generic-assert
    self.assertTrue(v1 == v2)
    self.assertTrue(v1 <= v2)
    self.assertTrue(v1 >= v2)
    self.assertTrue(v1 == v3)
    self.assertTrue(v1 <= v3)
    self.assertTrue(v1 >= v3)
    # pylint: enable=g-generic-assert

  def test_neq(self):
    v1 = version.Version('1.2.3')
    v2 = version.Version('1.2.4')
    v3 = '1.2.4'
    # pylint: disable=g-generic-assert
    self.assertTrue(v1 != v2)
    self.assertTrue(v1 != v3)
    # pylint: enable=g-generic-assert

  def test_less(self):
    v1 = version.Version('1.2.3')
    v2 = version.Version('1.2.4')
    v3 = '1.2.4'
    # pylint: disable=g-generic-assert
    self.assertTrue(v1 < v2)
    self.assertTrue(v1 <= v2)
    self.assertTrue(v1 < v3)
    self.assertTrue(v1 <= v3)
    # pylint: enable=g-generic-assert

  def test_sup(self):
    v1 = version.Version('1.2.3')
    v2 = version.Version('1.2.4')
    v3 = '1.2.4'
    # pylint: disable=g-generic-assert
    self.assertTrue(v2 > v1)
    self.assertTrue(v2 >= v1)
    self.assertTrue(v3 > v1)
    self.assertTrue(v3 >= v1)
    # pylint: enable=g-generic-assert

  def test_experiment_default(self):
    v = version.Version('1.2.3')
    self.assertFalse(v.implements(version.Experiment.DUMMY))

  def test_experiment_override(self):
    v = version.Version('1.2.3', experiments={version.Experiment.DUMMY: True})
    self.assertTrue(v.implements(version.Experiment.DUMMY))

  def test_hash(self):
    self.assertIn(
        version.Version('1.2.3'),
        {version.Version('1.2.3'), version.Version('1.4.3')}
    )

    self.assertNotIn(
        version.Version('1.2.3'),
        {version.Version('1.1.3'), version.Version('1.4.3')}
    )


def test_str_to_version():
  v0 = version.Version('1.2.3')
  v1 = version.Version(v0)
  assert v1 == v0


if __name__ == '__main__':
  testing.test_main()
