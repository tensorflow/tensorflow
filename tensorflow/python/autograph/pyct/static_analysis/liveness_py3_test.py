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
"""Tests for liveness module, that only run in Python 3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness_test
from tensorflow.python.platform import test


NodeAnno = annos.NodeAnno


class LivenessAnalyzerTest(liveness_test.LivenessAnalyzerTestBase):
  """Tests which can only run in Python 3."""

  def test_nonlocal_symbol(self):

    nonlocal_a = 3
    nonlocal_b = 13

    def test_fn(c):
      nonlocal nonlocal_a
      nonlocal nonlocal_b
      if nonlocal_a:
        nonlocal_b = c
      else:
        nonlocal_b = c
      return nonlocal_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body
    self.assertHasLiveOut(fn_body[2], ('nonlocal_b',))
    self.assertHasLiveIn(fn_body[2], ('nonlocal_a', 'c'))


if __name__ == '__main__':
  test.main()
