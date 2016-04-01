# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from tensorflow.tensorboard.lib.python import json_util

_INFINITY = float('inf')


class FloatWrapperTest(googletest.TestCase):

  def _assertWrapsAs(self, to_wrap, expected):
    """Asserts that |to_wrap| becomes |expected| when wrapped."""
    actual = json_util.WrapSpecialFloats(to_wrap)
    for a, e in zip(actual, expected):
      self.assertEqual(e, a)

  def testWrapsPrimitives(self):
    self._assertWrapsAs(_INFINITY, 'Infinity')
    self._assertWrapsAs(-_INFINITY, '-Infinity')
    self._assertWrapsAs(float('nan'), 'NaN')

  def testWrapsObjectValues(self):
    self._assertWrapsAs({'x': _INFINITY}, {'x': 'Infinity'})

  def testWrapsObjectKeys(self):
    self._assertWrapsAs({_INFINITY: 'foo'}, {'Infinity': 'foo'})

  def testWrapsInListsAndTuples(self):
    self._assertWrapsAs([_INFINITY], ['Infinity'])
    # map() returns a list even if the argument is a tuple.
    self._assertWrapsAs((_INFINITY,), ['Infinity',])

  def testWrapsRecursively(self):
    self._assertWrapsAs({'x': [_INFINITY]}, {'x': ['Infinity']})


if __name__ == '__main__':
  googletest.main()
