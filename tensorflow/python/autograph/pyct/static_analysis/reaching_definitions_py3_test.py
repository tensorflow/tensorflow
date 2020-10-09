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

  def test_nonlocal(self):

    a = 3
    b = 13

    def test_fn():
      nonlocal a
      nonlocal b
      if a:
        b = []
      return a, b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

  def test_nonlocal_in_nested_function(self):

    a = 3
    b = 13

    def test_fn():
      a = 3
      b = 13

      def local_fn():
        nonlocal a, b
        if a:
          b = []
        return a, b

      return local_fn()

    node = self._parse_and_analyze(test_fn)
    local_body = node.body[2].body

    self.assertHasDefs(local_body[1].test, 1)
    self.assertHasDefs(local_body[1].body[0].targets[0], 1)
    self.assertHasDefs(local_body[2].value.elts[0], 1)
    self.assertHasDefs(local_body[2].value.elts[1], 2)

    self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])

    # Note: the function name is is visible inside the function body. But it's
    # a closure variable, not a local.
    #
    # Example:
    #
    #   >>> def f():
    #   ...  print(f)
    #   >>> g = f
    #   >>> f = 'something else'
    #   >>> g()
    #   something else
    #
    self.assertHasDefinedIn(local_body[1], ('a', 'b'))


if __name__ == '__main__':
  test.main()
