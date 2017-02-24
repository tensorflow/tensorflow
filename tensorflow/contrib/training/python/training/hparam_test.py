# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for hparam."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.training.python.training import hparam

from tensorflow.python.platform import test


class HParamsTest(test.TestCase):

  def _assertDictEquals(self, d1, d2):
    self.assertEqual(len(d1), len(d2))
    for k, v in six.iteritems(d1):
      self.assertTrue(k in d2, k)
      self.assertEquals(v, d2[k], d2[k])

  def testEmpty(self):
    hparams = hparam.HParams()
    self._assertDictEquals({}, hparams.values())
    hparams.parse('')
    self._assertDictEquals({}, hparams.values())
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('xyz=123')

  def testSomeValues(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6')
    self._assertDictEquals(
        {'aaa': 1, 'b': 2.0, 'c_c': 'relu6'}, hparams.values())
    expected_str = '[(\'aaa\', 1), (\'b\', 2.0), (\'c_c\', \'relu6\')]'
    self.assertEquals(expected_str, str(hparams.__str__()))
    self.assertEquals(expected_str, str(hparams))
    self.assertEquals(1, hparams.aaa)
    self.assertEquals(2.0, hparams.b)
    self.assertEquals('relu6', hparams.c_c)
    hparams.parse('aaa=12')
    self._assertDictEquals(
        {'aaa': 12, 'b': 2.0, 'c_c': 'relu6'}, hparams.values())
    self.assertEquals(12, hparams.aaa)
    self.assertEquals(2.0, hparams.b)
    self.assertEquals('relu6', hparams.c_c)
    hparams.parse('c_c=relu4,b=-2.0e10')
    self._assertDictEquals({'aaa': 12, 'b': -2.0e10, 'c_c': 'relu4'},
                           hparams.values())
    self.assertEquals(12, hparams.aaa)
    self.assertEquals(-2.0e10, hparams.b)
    self.assertEquals('relu4', hparams.c_c)
    hparams.parse('c_c=,b=0,')
    self._assertDictEquals({'aaa': 12, 'b': 0, 'c_c': ''}, hparams.values())
    self.assertEquals(12, hparams.aaa)
    self.assertEquals(0.0, hparams.b)
    self.assertEquals('', hparams.c_c)
    hparams.parse('c_c=2.3",b=+2,')
    self.assertEquals(2.0, hparams.b)
    self.assertEquals('2.3"', hparams.c_c)
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('x=123')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=poipoi')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=1.0')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=12x')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=relu')
    with self.assertRaisesRegexp(ValueError, 'Must not pass a list'):
      hparams.parse('aaa=[123]')
    self.assertEquals(12, hparams.aaa)
    self.assertEquals(2.0, hparams.b)
    self.assertEquals('2.3"', hparams.c_c)
    # Exports to proto.
    hparam_def = hparams.to_proto()
    # Imports from proto.
    hparams2 = hparam.HParams(hparam_def=hparam_def)
    # Verifies that all hparams are restored.
    self.assertEquals(12, hparams2.aaa)
    self.assertEquals(2.0, hparams2.b)
    self.assertEquals('2.3"', hparams2.c_c)

  def testBoolParsing(self):
    for value in 'true', 'false', '1', '0':
      for initial in False, True:
        hparams = hparam.HParams(use_gpu=initial)
        hparams.parse('use_gpu=' + value)
        self.assertEqual(hparams.use_gpu, value in ['true', '1'])

        # Exports to proto.
        hparam_def = hparams.to_proto()
        # Imports from proto.
        hparams2 = hparam.HParams(hparam_def=hparam_def)
        self.assertEquals(hparams.use_gpu, hparams2.use_gpu)
        # Check that hparams2.use_gpu is a bool rather than an int.
        # The assertEquals() call above won't catch this, since
        # (0 == False) and (1 == True) in Python.
        self.assertEquals(bool, type(hparams2.use_gpu))

  def testBoolParsingFail(self):
    hparams = hparam.HParams(use_gpu=True)
    with self.assertRaisesRegexp(ValueError, r'Could not parse.*use_gpu'):
      hparams.parse('use_gpu=yep')

  def testLists(self):
    hparams = hparam.HParams(aaa=[1], b=[2.0, 3.0], c_c=['relu6'])
    self._assertDictEquals({'aaa': [1], 'b': [2.0, 3.0], 'c_c': ['relu6']},
                           hparams.values())
    self.assertEquals([1], hparams.aaa)
    self.assertEquals([2.0, 3.0], hparams.b)
    self.assertEquals(['relu6'], hparams.c_c)
    hparams.parse('aaa=[12]')
    self.assertEquals([12], hparams.aaa)
    hparams.parse('aaa=[12,34,56]')
    self.assertEquals([12, 34, 56], hparams.aaa)
    hparams.parse('c_c=[relu4,relu12],b=[1.0]')
    self.assertEquals(['relu4', 'relu12'], hparams.c_c)
    self.assertEquals([1.0], hparams.b)
    hparams.parse('c_c=[],aaa=[-34]')
    self.assertEquals([-34], hparams.aaa)
    self.assertEquals([], hparams.c_c)
    hparams.parse('c_c=[_12,3\'4"],aaa=[+3]')
    self.assertEquals([3], hparams.aaa)
    self.assertEquals(['_12', '3\'4"'], hparams.c_c)
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('x=[123]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=[poipoi]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=[1.0]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=[12x]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=[relu]')
    with self.assertRaisesRegexp(ValueError, 'Must pass a list'):
      hparams.parse('aaa=123')
    # Exports to proto.
    hparam_def = hparams.to_proto()
    # Imports from proto.
    hparams2 = hparam.HParams(hparam_def=hparam_def)
    # Verifies that all hparams are restored.
    self.assertEquals([3], hparams2.aaa)
    self.assertEquals([1.0], hparams2.b)
    self.assertEquals(['_12', '3\'4"'], hparams2.c_c)

  def testJson(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6', d=True)
    self._assertDictEquals(
        {'aaa': 1, 'b': 2.0, 'c_c': 'relu6', 'd': True}, hparams.values())
    self.assertEquals(1, hparams.aaa)
    self.assertEquals(2.0, hparams.b)
    self.assertEquals('relu6', hparams.c_c)
    hparams.parse_json('{"aaa": 12, "b": 3.0, "c_c": "relu4", "d": false}')
    self._assertDictEquals(
        {'aaa': 12, 'b': 3.0, 'c_c': 'relu4', 'd': False}, hparams.values())
    self.assertEquals(12, hparams.aaa)
    self.assertEquals(3.0, hparams.b)
    self.assertEquals('relu4', hparams.c_c)

    json_str = hparams.to_json()
    hparams2 = hparam.HParams(aaa=10, b=20.0, c_c='hello', d=False)
    hparams2.parse_json(json_str)
    self.assertEquals(12, hparams2.aaa)
    self.assertEquals(3.0, hparams2.b)
    self.assertEquals('relu4', hparams2.c_c)
    self.assertEquals(False, hparams2.d)

  def testNonProtoFails(self):
    with self.assertRaisesRegexp(AssertionError, ''):
      hparam.HParams(hparam_def=1)
    with self.assertRaisesRegexp(AssertionError, ''):
      hparam.HParams(hparam_def=1.0)
    with self.assertRaisesRegexp(AssertionError, ''):
      hparam.HParams(hparam_def='hello')
    with self.assertRaisesRegexp(AssertionError, ''):
      hparam.HParams(hparam_def=[1, 2, 3])


if __name__ == '__main__':
  test.main()
