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

from tensorflow.python.data.util import random_seed as data_random_seed
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class RandomSeedTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testRandomSeed(self):
    zero_t = constant_op.constant(0, dtype=dtypes.int64, name='zero')
    one_t = constant_op.constant(1, dtype=dtypes.int64, name='one')
    intmax_t = constant_op.constant(
        2**31 - 1, dtype=dtypes.int64, name='intmax')
    test_cases = [
        # Each test case is a tuple with input to get_seed:
        # (input_graph_seed, input_op_seed)
        # and output from get_seed:
        # (output_graph_seed, output_op_seed)
        ((None, None), (0, 0)),
        ((None, 1), (random_seed.DEFAULT_GRAPH_SEED, 1)),
        ((1, 1), (1, 1)),
        ((0, 0), (0, 2**31 - 1)),  # Avoid nondeterministic (0, 0) output
        ((2**31 - 1, 0), (0, 2**31 - 1)),  # Don't wrap to (0, 0) either
        ((0, 2**31 - 1), (0, 2**31 - 1)),  # Wrapping for the other argument
        # Once more, with tensor-valued arguments
        ((None, one_t), (random_seed.DEFAULT_GRAPH_SEED, 1)),
        ((1, one_t), (1, 1)),
        ((0, zero_t), (0, 2**31 - 1)),  # Avoid nondeterministic (0, 0) output
        ((2**31 - 1, zero_t), (0, 2**31 - 1)),  # Don't wrap to (0, 0) either
        ((0, intmax_t), (0, 2**31 - 1)),  # Wrapping for the other argument
    ]
    for tc in test_cases:
      tinput, toutput = tc[0], tc[1]
      random_seed.set_random_seed(tinput[0])
      g_seed, op_seed = data_random_seed.get_seed(tinput[1])
      g_seed = self.evaluate(g_seed)
      op_seed = self.evaluate(op_seed)
      msg = 'test_case = {0}, got {1}, want {2}'.format(
          tinput, (g_seed, op_seed), toutput)
      self.assertEqual((g_seed, op_seed), toutput, msg=msg)
      random_seed.set_random_seed(None)

    if context.in_graph_mode():
      random_seed.set_random_seed(1)
      tinput = (1, None)
      toutput = (1, ops.get_default_graph()._last_id)  # pylint: disable=protected-access
      random_seed.set_random_seed(tinput[0])
      g_seed, op_seed = data_random_seed.get_seed(tinput[1])
      g_seed = self.evaluate(g_seed)
      op_seed = self.evaluate(op_seed)
      msg = 'test_case = {0}, got {1}, want {2}'.format(1, (g_seed, op_seed),
                                                        toutput)
      self.assertEqual((g_seed, op_seed), toutput, msg=msg)
      random_seed.set_random_seed(None)


if __name__ == '__main__':
  test.main()
