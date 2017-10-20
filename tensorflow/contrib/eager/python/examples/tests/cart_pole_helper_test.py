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

from tensorflow.contrib.eager.python.examples import cart_pole_helper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class RewardDiscountingTest(test_util.TensorFlowTestCase):

  def testDiscountingRewards(self):
    rewards = [0.0, 10.0, 20.0]
    discount_rate = 0.9
    self.assertAllClose(
        [10 * discount_rate + 20 * discount_rate * discount_rate,
         10 + 20 * discount_rate, 20],
        cart_pole_helper.discount_rewards(rewards, discount_rate))
    self.assertAllClose(
        [-1.2], cart_pole_helper.discount_rewards([-1.2], discount_rate))
    self.assertEqual([], cart_pole_helper.discount_rewards([], discount_rate))

  def testDiscountAndNormalizeRewardSequences(self):
    rewards1 = [0.0, 10.0, 20.0]
    rewards2 = [0.0, 5.0, -5.0]
    reward_sequences = [rewards1, rewards2]
    discount_rate = 0.9
    dn = cart_pole_helper.discount_and_normalize_rewards(reward_sequences,
                                                         discount_rate)
    self.assertAllClose(
        [[1.03494653, 1.24685514, 0.64140196],
         [-0.83817424, -0.83439016, -1.25063922]], dn)


if __name__ == "__main__":
  test.main()
