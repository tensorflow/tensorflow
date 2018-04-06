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
"""Tests for the SeedStream class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import seed_stream
from tensorflow.python.platform import test


class SeedStreamTest(test.TestCase):

  def assertAllUnique(self, items):
    self.assertEqual(len(items), len(set(items)))

  def testNonRepetition(self):
    # The probability of repetitions in a short stream from a correct
    # PRNG is negligible; this test catches bugs that prevent state
    # updates.
    strm = seed_stream.SeedStream(seed=4, salt="salt")
    output = [strm() for _ in range(50)]
    self.assertEqual(sorted(output), sorted(list(set(output))))

  def testReproducibility(self):
    strm1 = seed_stream.SeedStream(seed=4, salt="salt")
    strm2 = seed_stream.SeedStream(seed=4, salt="salt")
    strm3 = seed_stream.SeedStream(seed=4, salt="salt")
    outputs = [strm1() for _ in range(50)]
    self.assertEqual(outputs, [strm2() for _ in range(50)])
    self.assertEqual(outputs, [strm3() for _ in range(50)])

  def testSeededDistinctness(self):
    strm1 = seed_stream.SeedStream(seed=4, salt="salt")
    strm2 = seed_stream.SeedStream(seed=5, salt="salt")
    self.assertAllUnique(
        [strm1() for _ in range(50)] + [strm2() for _ in range(50)])

  def testSaltedDistinctness(self):
    strm1 = seed_stream.SeedStream(seed=4, salt="salt")
    strm2 = seed_stream.SeedStream(seed=4, salt="another salt")
    self.assertAllUnique(
        [strm1() for _ in range(50)] + [strm2() for _ in range(50)])

  def testNestingRobustness(self):
    # SeedStreams started from generated seeds should not collide with
    # the master or with each other, even if the salts are the same.
    strm1 = seed_stream.SeedStream(seed=4, salt="salt")
    strm2 = seed_stream.SeedStream(strm1(), salt="salt")
    strm3 = seed_stream.SeedStream(strm1(), salt="salt")
    outputs = [strm1() for _ in range(50)]
    self.assertAllUnique(
        outputs + [strm2() for _ in range(50)] + [strm3() for _ in range(50)])


if __name__ == "__main__":
  test.main()
