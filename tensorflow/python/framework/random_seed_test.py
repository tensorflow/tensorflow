# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class RandomSeedTest(test.TestCase):

  def testRandomSeed(self):
    test_cases = [
        # Each test case is a tuple with input to get_seed:
        # (input_graph_seed, input_op_seed)
        # and output from get_seed:
        # (output_graph_seed, output_op_seed)
        ((None, None), (None, None)),
        ((None, 1), (random_seed.DEFAULT_GRAPH_SEED, 1)),
        ((1, None), (1, 0)),  # 0 will be the default_graph._lastid.
        ((1, 1), (1, 1)),
    ]
    for tc in test_cases:
      tinput, toutput = tc[0], tc[1]
      random_seed.set_random_seed(tinput[0])
      g_seed, op_seed = random_seed.get_seed(tinput[1])
      msg = 'test_case = {0}, got {1}, want {2}'.format(tinput,
                                                        (g_seed, op_seed),
                                                        toutput)
      self.assertEqual((g_seed, op_seed), toutput, msg=msg)
      random_seed.set_random_seed(None)


if __name__ == '__main__':
  test.main()
