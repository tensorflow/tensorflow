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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import metrics
from tensorflow.python.eager import test


class MetricsTest(test.TestCase):

  def testMean(self):
    m = metrics.Mean()
    m([1, 10, 100])
    m(1000)
    m([10000.0, 100000.0])
    self.assertEqual(111111.0/6, m.result().numpy())

  def testWeightedMean(self):
    m = metrics.Mean()
    m([1, 100, 100000], weights=[1, 0.2, 0.3])
    m([500000, 5000, 500])  # weights of 1 each
    self.assertNear(535521/4.5, m.result().numpy(), 0.001)

  def testAccuracy(self):
    m = metrics.Accuracy()
    m([0, 1, 2, 3], [0, 0, 0, 0])  # 1 correct
    m([4], [4])  # 1 correct
    m([5], [0])  # 0 correct
    m([6], [6])  # 1 correct
    m([7], [2])  # 0 correct
    self.assertEqual(3.0/8, m.result().numpy())

  def testWeightedAccuracy(self):
    m = metrics.Accuracy()
    # 1 correct, total weight of 2
    m([0, 1, 2, 3], [0, 0, 0, 0], weights=[1, 1, 0, 0])
    m([4], [4], weights=[0.5])  # 1 correct with a weight of 0.5
    m([5], [0], weights=[0.5])  # 0 correct, weight 0.5
    m([6], [6])  # 1 correct, weight 1
    m([7], [2])  # 0 correct, weight 1
    self.assertEqual(2.5/5, m.result().numpy())


if __name__ == "__main__":
  test.main()
