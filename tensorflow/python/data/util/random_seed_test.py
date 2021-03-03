# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utilities working with arbitrarily nested structures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import random_seed as data_random_seed
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


# NOTE(vikoth18): Arguments of parameterized tests are lifted into lambdas to make
# sure they are not executed before the (eager- or graph-mode) test environment
# has been set up.


def _test_random_seed_combinations():

  cases = [
      # Each test case is a tuple with input to get_seed:
      # (input_graph_seed, input_op_seed)
      # and output from get_seed:
      # (output_graph_seed, output_op_seed)
      (
          "TestCase_0",
          lambda: (None, None),
          lambda: (0, 0),
      ),
      ("TestCase_1", lambda: (None, 1), lambda:
       (random_seed.DEFAULT_GRAPH_SEED, 1)),
      ("TestCase_2", lambda: (1, 1), lambda: (1, 1)),
      (
          # Avoid nondeterministic (0, 0) output
          "TestCase_3",
          lambda: (0, 0),
          lambda: (0, 2**31 - 1)),
      (
          # Don't wrap to (0, 0) either
          "TestCase_4",
          lambda: (2**31 - 1, 0),
          lambda: (0, 2**31 - 1)),
      (
          # Wrapping for the other argument
          "TestCase_5",
          lambda: (0, 2**31 - 1),
          lambda: (0, 2**31 - 1)),
      (
          # Once more, with tensor-valued arguments
          "TestCase_6",
          lambda:
          (None, constant_op.constant(1, dtype=dtypes.int64, name="one")),
          lambda: (random_seed.DEFAULT_GRAPH_SEED, 1)),
      ("TestCase_7", lambda:
       (1, constant_op.constant(1, dtype=dtypes.int64, name="one")), lambda:
       (1, 1)),
      (
          "TestCase_8",
          lambda: (0, constant_op.constant(0, dtype=dtypes.int64, name="zero")),
          lambda: (0, 2**31 - 1)  # Avoid nondeterministic (0, 0) output
      ),
      (
          "TestCase_9",
          lambda:
          (2**31 - 1, constant_op.constant(0, dtype=dtypes.int64, name="zero")),
          lambda: (0, 2**31 - 1)  # Don't wrap to (0, 0) either
      ),
      (
          "TestCase_10",
          lambda:
          (0, constant_op.constant(
              2**31 - 1, dtype=dtypes.int64, name="intmax")),
          lambda: (0, 2**31 - 1)  # Wrapping for the other argument
      )
  ]

  def reduce_fn(x, y):
    name, input_fn, output_fn = y
    return x + combinations.combine(
        input_fn=combinations.NamedObject("input_fn.{}".format(name), input_fn),
        output_fn=combinations.NamedObject("output_fn.{}".format(name),
                                           output_fn))

  return functools.reduce(reduce_fn, cases, [])


class RandomSeedTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _checkEqual(self, tinput, toutput):
    random_seed.set_random_seed(tinput[0])
    g_seed, op_seed = data_random_seed.get_seed(tinput[1])
    g_seed = self.evaluate(g_seed)
    op_seed = self.evaluate(op_seed)
    msg = "test_case = {0}, got {1}, want {2}".format(tinput, (g_seed, op_seed),
                                                      toutput)
    self.assertEqual((g_seed, op_seed), toutput, msg=msg)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_random_seed_combinations()))
  def testRandomSeed(self, input_fn, output_fn):
    tinput, toutput = input_fn(), output_fn()
    self._checkEqual(tinput=tinput, toutput=toutput)
    random_seed.set_random_seed(None)

  @combinations.generate(test_base.graph_only_combinations())
  def testIncrementalRandomSeed(self):
    random_seed.set_random_seed(1)
    for i in range(10):
      tinput = (1, None)
      toutput = (1, i)
      self._checkEqual(tinput=tinput, toutput=toutput)


if __name__ == '__main__':
  test.main()
