# python3
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
"""Tests for reaching_definitions module, that only run in Python 3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions_test
from tensorflow.python.platform import test


class ReachingDefinitionsAnalyzerTest(
    reaching_definitions_test.ReachingDefinitionsAnalyzerTestBase):
  """Tests which can only run in Python 3."""

  def test_nonlocal_symbol(self):

    nonlocal_a = 3
    nonlocal_b = 13

    def test_fn():
      nonlocal nonlocal_a
      nonlocal nonlocal_b
      if nonlocal_a:
        nonlocal_b = []
      return nonlocal_a, nonlocal_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('nonlocal_a', 'nonlocal_b'))


if __name__ == '__main__':
  test.main()
