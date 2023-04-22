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
"""Tests for misc module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.utils import misc
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops.variables import Variable
from tensorflow.python.platform import test


class MiscTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_alias_single_tensor(self):
    a = constant(1)

    new_a = misc.alias_tensors(a)
    self.assertFalse(new_a is a)
    with self.cached_session() as sess:
      self.assertEqual(1, self.evaluate(new_a))

  @test_util.run_deprecated_v1
  def test_alias_tensors(self):
    a = constant(1)
    v = Variable(2)
    s = 'a'
    l = [1, 2, 3]

    new_a, new_v, new_s, new_l = misc.alias_tensors(a, v, s, l)

    self.assertFalse(new_a is a)
    self.assertTrue(new_v is v)
    self.assertTrue(new_s is s)
    self.assertTrue(new_l is l)
    with self.cached_session() as sess:
      self.assertEqual(1, self.evaluate(new_a))

  def test_get_range_len(self):
    get_range_as_graph = def_function.function(misc.get_range_len)
    test_range = [(i, constant_op.constant(i)) for i in range(-3, 3)]
    results = []
    for i, ti in test_range:
      for j, tj in test_range:
        for k, tk in test_range:
          if k == 0:
            continue
          results.append(((i, j, k), get_range_as_graph(ti, tj, tk)))

    for (i, j, k), result_tensor in results:
      self.assertEqual(
          len(list(range(i, j, k))), self.evaluate(result_tensor))


if __name__ == '__main__':
  test.main()
